#include "../mean_pool.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace image_ops {
    namespace {
        template<typename scalar_t>
        __global__ void mean_pool_kernel(
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output,
            uint32_t kernel_size
        ) {
            const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if ((x >= output.size(0)) || (y >= output.size(1))) {
                return;
            }

            scalar_t pixel_sum = 0.0f;
            scalar_t area = kernel_size * kernel_size;

            #pragma unroll
            for (uint32_t i = 0; i < kernel_size; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < kernel_size; j++) {
                    pixel_sum += input[x * kernel_size + i][y * kernel_size + j];
                }
            }
            output[x][y] = pixel_sum / area;
        }
    };

    torch::Tensor mean_pool(torch::Tensor input, uint32_t kernel_size) {
        CHECK_INPUT(input);
        uint32_t H = input.size(0);
        uint32_t W = input.size(1);
        uint32_t downscaled_H = (H + kernel_size - 1) / kernel_size;
        uint32_t downscaled_W = (W + kernel_size - 1) / kernel_size;
        torch::Tensor output = torch::empty({downscaled_H, downscaled_W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        mean_pool_inplace(input, output, kernel_size);
        return output;
    }

    void mean_pool_inplace(torch::Tensor input, torch::Tensor output, uint32_t kernel_size) {
        CHECK_INPUT(input);
        CHECK_INPUT(output);

        const dim3 threads_per_block(16, 16);
        const dim3 num_blocks(
            (output.size(0) + threads_per_block.x - 1) / threads_per_block.x,
            (output.size(1) + threads_per_block.y - 1) / threads_per_block.y
        );

        AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_pool_inplace", ([&] {
            mean_pool_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
                input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                kernel_size
            );
        }));
    }
};