#include "ncc_matching_cost_volume_construction.hh"

#include "device_functions.cuh"

#include <limits>

namespace {
    template<typename scalar_t>
    __global__ void secondary_matching_kernel(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_left,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_right,
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cost_volume,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> downscaled_disparity,
        int32_t patch_radius,
        int32_t k
    ) {
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x >= downscaled_disparity.size(0)) || (y >= downscaled_disparity.size(1))) {
            return;
        }
        
        int32_t d_mbm = static_cast<int32_t>(downscaled_disparity[x][y]);
        int32_t d_mbm_minus_one = k * (d_mbm - 1);
        int32_t d_mbm_plus_one = k * (d_mbm + 1);

        const auto read_mbm_disparity_cost = [&](int32_t disparity) -> scalar_t {
            int32_t d_index = device_functions::pad_index(disparity, cost_volume.size(2));
            return cost_volume[x][y][d_index];
        };

        const auto read_sad_disparity_cost = [&](int32_t disparity) -> scalar_t {
            return device_functions::compute_sad_cost_function<scalar_t>(
                input_left,
                input_right,
                x * k,
                y * k,
                disparity,
                patch_radius,
                static_cast<scalar_t>(255)
            );
        };

        scalar_t c_sad = std::numeric_limits<scalar_t>::min();
        int32_t d_sad = d_mbm_minus_one;
        for (int32_t sad_disparity = d_mbm_minus_one; sad_disparity <= d_mbm_plus_one; sad_disparity++) {
            scalar_t sad_cost = read_sad_disparity_cost(sad_disparity);
            if (sad_cost > c_sad) {
                d_sad = sad_disparity;
                c_sad = sad_cost;
            }
        }

        if ((d_sad > d_mbm_minus_one) && (d_sad < d_mbm_plus_one)) {
            scalar_t mbm_quadratic_min = device_functions::quadratic_function_peak<scalar_t>(d_mbm, read_mbm_disparity_cost(d_mbm),
                                                                                             d_mbm + 1, read_mbm_disparity_cost(d_mbm + 1),
                                                                                             d_mbm - 1, read_mbm_disparity_cost(d_mbm - 1));
            scalar_t sad_quadratic_min = device_functions::quadratic_function_peak<scalar_t>(d_sad, c_sad,
                                                                                             d_sad + 1, read_sad_disparity_cost(d_sad + 1),
                                                                                             d_sad - 1, read_sad_disparity_cost(d_sad - 1));

            scalar_t delta_mbm = mbm_quadratic_min - d_mbm;
            scalar_t delta_sad = sad_quadratic_min - d_sad;

            if (device_functions::have_same_sign(delta_mbm, d_sad + delta_sad - k * d_mbm)) {
                downscaled_disparity[x][y] = (d_sad + delta_sad) / k;
            } else {
                downscaled_disparity[x][y] = (d_mbm + delta_mbm + ((d_sad + delta_sad) / k)) / 2;
            }
        }
    }
};

void secondary_matching_cuda(
    torch::Tensor input_left,
    torch::Tensor input_right,
    torch::Tensor cost_volume,
    torch::Tensor downscaled_disparity,
    int32_t patch_radius,
    int32_t k
) {
    const dim3 threads_per_block(16, 16);
    const dim3 num_blocks(
        (cost_volume.size(0) + threads_per_block.x - 1) / threads_per_block.x,
        (cost_volume.size(1) + threads_per_block.y - 1) / threads_per_block.y
    );

    AT_DISPATCH_FLOATING_TYPES(cost_volume.type(), "secondary_matching_cuda", ([&] {
        secondary_matching_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input_left.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            input_right.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            cost_volume.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            downscaled_disparity.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            patch_radius,
            k
        );
    }));
}