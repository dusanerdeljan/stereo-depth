#include "ncc_matching_cost_volume_construction.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

namespace {
    template<typename scalar_t>
    __global__ void horizontal_disparity_fill_cuda_kernel(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_left,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> upscaled_disparity,
        int32_t k,
        int32_t threshold
    ) {
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if ((x >= upscaled_disparity.size(0)) || (y >= upscaled_disparity.size(1))) {
            return;
        }

        int32_t mod = y % k;
        int32_t nearest_k = y - mod;

        scalar_t prev_disparity = upscaled_disparity[x][nearest_k];
        scalar_t next_disparity = upscaled_disparity[x][nearest_k + k];

        if (std::abs(prev_disparity - next_disparity) <= threshold) {
            upscaled_disparity[x][y] = prev_disparity + mod * (next_disparity - prev_disparity) / k;
        } else {
            scalar_t prev_color = input_left[x][nearest_k];
            scalar_t next_color = input_left[x][nearest_k + k];
            scalar_t current_color = input_left[x][y];
            if (std::abs(current_color - prev_color) <= std::abs(current_color - next_color)) {
                upscaled_disparity[x][y] = prev_disparity;
            } else {
                upscaled_disparity[x][y] = next_disparity;
            }
        }
    }
};

void horizontal_disparity_fill_cuda(
    torch::Tensor input_left,
    torch::Tensor upscaled_disparity,
    int32_t k,
    int32_t threshold
) {
    const dim3 threads_per_block(16, 16);
    const dim3 num_blocks(
        (upscaled_disparity.size(0) + threads_per_block.x - 1) / threads_per_block.x,
        (upscaled_disparity.size(1) + threads_per_block.y - 1) / threads_per_block.y
    );

    AT_DISPATCH_FLOATING_TYPES(upscaled_disparity.type(), "horizontal_disparity_fill_cuda", ([&] {
        horizontal_disparity_fill_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input_left.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            upscaled_disparity.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            k,
            threshold
        );
    }));
}