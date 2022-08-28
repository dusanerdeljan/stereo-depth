#include "ncc_matching_cost_volume_construction.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

namespace {
    template<typename scalar_t>
    __global__ void wta_disparity_selection_kernel(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cost_volume,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> downscaled_disparity,
        int32_t min_disparity
    ) {
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if ((x >= downscaled_disparity.size(0)) || (y >= downscaled_disparity.size(1))) {
            return;
        }

        scalar_t best_cost = std::numeric_limits<scalar_t>::min();
        scalar_t best_disparity = 0.0f;
        for (int32_t disparity = 0; disparity < cost_volume.size(2); disparity++) {
            if (cost_volume[x][y][disparity] > best_cost) {
                best_cost = cost_volume[x][y][disparity];
                best_disparity = disparity;
            }
        }
        downscaled_disparity[x][y] = best_disparity + min_disparity;
    }
};

void wta_disparity_selection_cuda(
    torch::Tensor cost_volume,
    torch::Tensor downscaled_disparity,
    int32_t min_disparity
) {
    const dim3 threads_per_block(8, 8);
    const dim3 num_blocks(
        (cost_volume.size(0) + threads_per_block.x - 1) / threads_per_block.x,
        (cost_volume.size(1) + threads_per_block.y - 1) / threads_per_block.y
    );

    AT_DISPATCH_FLOATING_TYPES(cost_volume.type(), "wta_disparity_selection_cuda", ([&] {
        wta_disparity_selection_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            cost_volume.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            downscaled_disparity.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            min_disparity
        );
    }));
}