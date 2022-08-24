#include "../rgb_to_grayscale.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace image_ops {
    namespace {
        template<typename scalar_t>
        __global__ void rgb_to_grayscale_kernel(
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output
        ) {
            const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if ((x >= output.size(0)) || (y >= output.size(1))) {
                return;
            }

            scalar_t R = 0.2989f * input[0][x][y];
            scalar_t G = 0.5870f * input[1][x][y];
            scalar_t B = 0.1140f * input[2][x][y];

            output[x][y] = R + G + B;
        }
    };

    torch::Tensor rgb_to_grayscale(torch::Tensor input) {
        CHECK_INPUT(input);
        torch::Tensor output = torch::empty({input.size(1), input.size(2)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        rgb_to_grayscale_inplace(input, output);
        return output;
    }

    void rgb_to_grayscale_inplace(torch::Tensor input, torch::Tensor output) {
        CHECK_INPUT(input);
        CHECK_INPUT(output);

        const dim3 threads_per_block(16, 16);
        const dim3 num_blocks(
            (output.size(0) + threads_per_block.x - 1) / threads_per_block.x,
            (output.size(1) + threads_per_block.y - 1) / threads_per_block.y
        );

        AT_DISPATCH_FLOATING_TYPES(input.type(), "rgb_to_grayscale_inplace", ([&] {
            rgb_to_grayscale_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
                input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        }));
    }
};