#include "ncc_matching_cost_volume_construction.hh"

#include "device_functions.cuh"

namespace {
    template<typename scalar_t>
    __global__ void ncc_matching_cost_volume_construction_kernel(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_left,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_right,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cost_volume,
        int32_t patch_radius,
        int32_t min_disparity,
        int32_t max_disparity
    ) {
        const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        const int32_t d = blockIdx.z * blockDim.z + threadIdx.z;
        const int32_t disparity = min_disparity + d;
        
        if ((x >= cost_volume.size(0)) || (y >= cost_volume.size(1)) || (disparity > max_disparity)) {
            return;
        }

        /*
        scalar_t cost = 0.0f;
        scalar_t left_sum = 0.0f;
        scalar_t right_sum = 0.0f;
        scalar_t left_sum_squared = 0.0f;
        scalar_t right_sum_squared = 0.0f;
        scalar_t patch_area = (2 * patch_radius + 1) * (2 * patch_radius + 1);

        // TODO: Read data into shared memory
        for (int32_t i = -patch_radius; i <= patch_radius; i++) {
            for (int32_t j = -patch_radius; j <= patch_radius; j++) {
                int32_t xx = device_functions::pad_index(x + i, cost_volume.size(0));
                int32_t yy = device_functions::pad_index(y + j, cost_volume.size(1));

                scalar_t left = input_left[xx][yy];
                left_sum += left;
                left_sum_squared += left * left;

                int32_t disp_index = device_functions::pad_index(yy - disparity, cost_volume.size(1));
                scalar_t right = input_left[xx][disp_index];
                right_sum += right;
                right_sum_squared += right * right;
            }
        }

        scalar_t left_mean = left_sum / patch_area;
        scalar_t right_mean = right_sum / patch_area;

        scalar_t left_stdev = sqrt((left_sum_squared - left_mean * left_mean) / patch_area);
        scalar_t right_stdev = sqrt((right_sum_squared - right_mean * right_mean) / patch_area);

        for (int32_t i = -patch_radius; i <= patch_radius; i++) {
            for (int32_t j = -patch_radius; j <= patch_radius; j++) {
                int32_t xx = device_functions::pad_index(x + i, cost_volume.size(0));
                int32_t yy = device_functions::pad_index(y + j, cost_volume.size(1));
                int32_t disp_index = device_functions::pad_index(yy - disparity, cost_volume.size(1));
                cost += (input_left[xx][yy] - left_mean) * (input_right[xx][disp_index] - right_mean);
            }
        }

        scalar_t denominator = patch_area * left_stdev * right_stdev;
        scalar_t total_cost = denominator != 0 ? (cost / denominator) : 0.0f;
        */
        scalar_t total_cost = device_functions::compute_sad_cost_function<scalar_t>(
            input_left,
            input_right,
            x,
            y,
            disparity,
            patch_radius,
            static_cast<scalar_t>(255)
        );
        cost_volume[x][y][d] = total_cost;
    }
};

void ncc_matching_cost_volume_construction_cuda(
    torch::Tensor input_left,
    torch::Tensor input_right,
    torch::Tensor cost_volume,
    int32_t patch_radius,
    int32_t min_disparity,
    int32_t max_disparity
) {
    const dim3 threads_per_block(8, 8, 1);
    const dim3 num_blocks(
        (cost_volume.size(0) + threads_per_block.x - 1) / threads_per_block.x,
        (cost_volume.size(1) + threads_per_block.y - 1) / threads_per_block.y,
        max_disparity - min_disparity + 1
    );

    AT_DISPATCH_FLOATING_TYPES(cost_volume.type(), "ncc_matching_cost_volume_construction", ([&] {
        ncc_matching_cost_volume_construction_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input_left.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            input_right.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            cost_volume.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            patch_radius,
            min_disparity,
            max_disparity
        );
    }));
}