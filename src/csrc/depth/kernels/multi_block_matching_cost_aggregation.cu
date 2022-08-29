#include "ncc_matching_cost_volume_construction.hh"

#include "device_functions.cuh"

namespace {
    template<typename scalar_t>
    __global__ void multi_block_matching_cost_aggregation_kernel(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cost_volume,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> block_cost_volume,
        int32_t small_radius,
        int32_t mid_radius,
        int32_t large_radius,
        uint32_t shared_height,
        uint32_t shared_width
    ) {
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t d = blockIdx.z * blockDim.z + threadIdx.z;
        
        if ((x >= cost_volume.size(0)) || (y >= cost_volume.size(1)) || (d >= cost_volume.size(2))) {
            return;
        }

        const auto pad_x = [&](int32_t x) -> int32_t {
            return device_functions::pad_index(x, cost_volume.size(0));
        };

        const auto pad_y = [&](int32_t y) -> int32_t {
            return device_functions::pad_index(y, cost_volume.size(1));
        };
        
        const int32_t start_x = blockDim.x * blockIdx.x - large_radius;
        const int32_t start_y = blockDim.y * blockIdx.y - large_radius;
        
        // https://stackoverflow.com/a/27570775
        extern __shared__ __align__(sizeof(scalar_t)) unsigned char raw_shared_mem_buffer[];
        scalar_t* shared_mem = reinterpret_cast<scalar_t*>(raw_shared_mem_buffer);
        
        const uint32_t read_per_x = shared_height / blockDim.x;
        const uint32_t read_per_y = shared_width / blockDim.y;

        for (uint32_t i = 0; i < read_per_x; i++) {
            for (uint32_t j = 0; j < read_per_y; j++) {
                const int32_t x_index = pad_x(start_x + threadIdx.x * read_per_x + i);
                const int32_t y_index = pad_y(start_y + threadIdx.y * read_per_y + j);

                shared_mem[threadIdx.y * read_per_y + j + (threadIdx.x * read_per_x + i) * shared_width] = cost_volume[x_index][y_index][d];
            }
        }

        __syncthreads();

        // Remap x, y to shared memory index space
        const int32_t x_remap = large_radius + threadIdx.x;
        const int32_t y_remap = large_radius + threadIdx.y;

        // Horizontal line block cost
        scalar_t horizontal_cost = 0.0f;
        for (int32_t i = -small_radius; i <= small_radius; i++) {
            for (int32_t j = -large_radius; j <= large_radius; j++) {
                int32_t x_index = x_remap + i;
                int32_t y_index = y_remap + j;
                horizontal_cost += shared_mem[y_index + x_index * shared_width];
            }
        }

        // Vertical line block cost
        scalar_t vertical_cost = 0.0f;
        for (int32_t i = -large_radius; i <= large_radius; i++) {
            for (int32_t j = -small_radius; j <= small_radius; j++) {
                int32_t x_index = x_remap + i;
                int32_t y_index = y_remap + j;
                vertical_cost += shared_mem[y_index + x_index * shared_width];
            }
        }

        // Center cross block cost
        scalar_t cross_cost = 0.0f;
        for (int32_t i = -mid_radius; i <= mid_radius; i++) {
            for (int32_t j = -mid_radius; j <= mid_radius; j++) {
                int32_t x_index = x_remap + i;
                int32_t y_index = y_remap + j;
                cross_cost += shared_mem[y_index + x_index * shared_width];
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
    const dim3 threads_per_block(4, 8, 1);
    const dim3 num_blocks(
        (cost_volume.size(0) + threads_per_block.x - 1) / threads_per_block.x,
        (cost_volume.size(1) + threads_per_block.y - 1) / threads_per_block.y,
        max_disparity - min_disparity + 1
    );

    const uint32_t shared_height = (3 * threads_per_block.x + 2 * large_radius - 1);
    const uint32_t shared_width = (3 * threads_per_block.y + 2 * large_radius - 1);

    AT_DISPATCH_FLOATING_TYPES(cost_volume.type(), "multi_block_matching_cost_aggregation_cuda", ([&] {
        multi_block_matching_cost_aggregation_kernel<scalar_t><<<num_blocks, threads_per_block, shared_height * shared_width * sizeof(scalar_t)>>>(
            cost_volume.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            block_cost_volume.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            small_radius,
            mid_radius,
            large_radius,
            shared_height,
            shared_width
        );
    }));
}