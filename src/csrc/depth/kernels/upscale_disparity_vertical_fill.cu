#include "ncc_matching_cost_volume_construction.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

namespace {
    template<typename scalar_t>
    __global__ void upscale_disparity_vertical_fill_kernel(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_left,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> downscaled_disparity,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> upscaled_disparity,
        int32_t k,
        int32_t threshold
    ) {
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if ((k * x >= upscaled_disparity.size(0)) || (k * y >= upscaled_disparity.size(1))) {
            return;
        }

        upscaled_disparity[k * x][k * y] = k * downscaled_disparity[x][y];
        
        if (x == 0) {
            return;
        }

        scalar_t prev_color = input_left[k * x][k * y];
        scalar_t next_color = input_left[(k + 1) * x][k * y];

        scalar_t prev_disparity = k * downscaled_disparity[x][y];
        scalar_t next_disparity = k * downscaled_disparity[x - 1][y];

        if (std::abs(prev_disparity - next_disparity) <= threshold) {
            #pragma unroll
            for (int32_t i = 1; i < k; i++) {
                upscaled_disparity[k * x + i][k * y] = prev_disparity + i * (next_disparity - prev_disparity) / k;
            }
        } else {
            #pragma unroll
            for (int32_t i = 1; i < k; i++) {
                scalar_t current_color = input_left[k * x + i][k * y];
                if (std::abs(current_color - prev_color) <= std::abs(current_color - next_color)) {
                    upscaled_disparity[k * x + i][k * y] = prev_disparity;
                } else {
                    upscaled_disparity[k * x + i][k * y] = next_disparity;
                }
            }
        }
    }
};

void upscale_disparity_vertical_fill_cuda(
    torch::Tensor input_left,
    torch::Tensor downscaled_disparity,
    torch::Tensor upscaled_disparity,
    int32_t k,
    int32_t threshold
) {
    const dim3 threads_per_block(4, 8);
    const dim3 num_blocks(
        (downscaled_disparity.size(0) + threads_per_block.x - 1) / threads_per_block.x,
        (downscaled_disparity.size(1) + threads_per_block.y - 1) / threads_per_block.y
    );

    AT_DISPATCH_FLOATING_TYPES(downscaled_disparity.type(), "upscale_disparity_vertical_fill_cuda", ([&] {
        upscale_disparity_vertical_fill_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input_left.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            downscaled_disparity.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            upscaled_disparity.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            k,
            threshold
        );
    }));
}