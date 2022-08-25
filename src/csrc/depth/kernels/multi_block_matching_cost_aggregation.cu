#include "ncc_matching_cost_volume_construction.hh"

#include "device_functions.cuh"

namespace {
    template<typename scalar_t>
    __global__ void multi_block_matching_cost_aggregation_kernel(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cost_volume,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> block_cost_volume,
        int32_t small_radius,
        int32_t mid_radius,
        int32_t large_radius
    ) {
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t d = blockIdx.z * blockDim.z + threadIdx.z;
        
        if ((x >= cost_volume.size(0)) || (y >= cost_volume.size(1)) || (d >= cost_volume.size(2))) {
            return;
        }

        // Horizontal line block cost
        scalar_t horizontal_cost = 0.0f;
        for (int32_t i = -small_radius; i <= small_radius; i++) {
            for (int32_t j = -large_radius; j <= large_radius; j++) {
                int32_t x_index = device_functions::pad_index(x + i, cost_volume.size(0));
                int32_t y_index = device_functions::pad_index(y + j, cost_volume.size(1));
                horizontal_cost += cost_volume[x_index][y_index][d];
            }
        }

        // Vertical line block cost
        scalar_t vertical_cost = 0.0f;
        for (int32_t i = -large_radius; i <= large_radius; i++) {
            for (int32_t j = -small_radius; j <= small_radius; j++) {
                int32_t x_index = device_functions::pad_index(x + i, cost_volume.size(0));
                int32_t y_index = device_functions::pad_index(y + j, cost_volume.size(1));
                vertical_cost += cost_volume[x_index][y_index][d];
            }
        }

        // Center cross block cost
        scalar_t cross_cost = 0.0f;
        for (int32_t i = -mid_radius; i <= mid_radius; i++) {
            for (int32_t j = -mid_radius; j <= mid_radius; j++) {
                int32_t x_index = device_functions::pad_index(x + i, cost_volume.size(0));
                int32_t y_index = device_functions::pad_index(y + j, cost_volume.size(1));
                cross_cost += cost_volume[x_index][y_index][d];
            }
        }

        scalar_t total_cost = horizontal_cost * vertical_cost * cross_cost;
        block_cost_volume[x][y][d] = total_cost;
    }
};

void multi_block_matching_cost_aggregation_cuda(
    torch::Tensor cost_volume,
    torch::Tensor block_cost_volume,
    int32_t min_disparity,
    int32_t max_disparity,
    int32_t small_radius,
    int32_t mid_radius,
    int32_t large_radius
) {
    const dim3 threads_per_block(16, 16, 1);
    const dim3 num_blocks(
        (cost_volume.size(0) + threads_per_block.x - 1) / threads_per_block.x,
        (cost_volume.size(1) + threads_per_block.y - 1) / threads_per_block.y,
        max_disparity - min_disparity + 1
    );

    AT_DISPATCH_FLOATING_TYPES(cost_volume.type(), "multi_block_matching_cost_aggregation_cuda", ([&] {
        multi_block_matching_cost_aggregation_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            cost_volume.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            block_cost_volume.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            small_radius,
            mid_radius,
            large_radius
        );
    }));
}