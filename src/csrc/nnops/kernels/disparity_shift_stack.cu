#include "disparity_shift_stack.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace nn_ops {
    namespace {
        template<typename scalar_t>
        __global__ void disparity_shift_stack_kernel(
            const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> left_image,
            torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> outupt,
            uint32_t N,
            uint32_t C,
            uint32_t H,
            uint32_t W,
            int32_t min_disparity
        ) {
            const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            const int32_t d = min_disparity + blockIdx.z;
    
            if ((x >= H) || (y + d >= W)) {
                return;
            }
            
            #pragma unroll
            for (uint32_t n = 0; n < N; n++) {
                #pragma unroll
                for (uint32_t c = 0; c < C; c++) {
                    outupt[n][d][c][x][y] = left_image[n][c][x][y + d];
                }
            }
        }
    };
    
    torch::Tensor disparity_shift_stack(
        torch::Tensor left_image,
        int32_t min_disparity,
        int32_t max_disparity
    ) {
        CHECK_INPUT(left_image);

        const uint32_t disparity_range = std::abs(max_disparity - min_disparity) + 1;
        const uint32_t N = left_image.size(0);
        const uint32_t C = left_image.size(1);
        const uint32_t H = left_image.size(2);
        const uint32_t W = left_image.size(3);

        torch::Tensor stacked_shifted_views = torch::zeros({N, disparity_range, C, H, W}, 
                                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        const dim3 threads_per_block(16, 16, 1);
        const dim3 num_blocks(
            (H + threads_per_block.x - 1) / threads_per_block.x,
            (W + threads_per_block.y - 1) / threads_per_block.y,
            disparity_range
        );
    
        AT_DISPATCH_FLOATING_TYPES(left_image.type(), "disparity_shift_stack", ([&] {
            disparity_shift_stack_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
                left_image.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                stacked_shifted_views.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
                N,
                C,
                H,
                W,
                min_disparity
            );
        }));

        return stacked_shifted_views;
    }
};