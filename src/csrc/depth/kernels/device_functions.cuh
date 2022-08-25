#pragma once

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace device_functions {

    __device__ __forceinline__ int32_t pad_index(int32_t index, int32_t dim_size) {
        if ((index >= 0) && (index < dim_size)) {
            return index;
        } else if (index < 0) {
            return dim_size + index;
        } else if (index == dim_size) {
            return 0;
        } else {
            return dim_size - index;
        }
    } 

    template<typename scalar_t>
    __device__ scalar_t compute_sad_cost_function(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_left,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_right,
        int32_t x,
        int32_t y,
        int32_t disparity,
        int32_t patch_radius,
        scalar_t max_val
    ) {
        scalar_t total_cost = 0.0f;
        for (int32_t i = -patch_radius; i <= patch_radius; i++) {
            for (int32_t j = -patch_radius; j <= patch_radius; j++) {
                int32_t x_index = pad_index(x + i, input_left.size(0));
                int32_t y_index = pad_index(y + j, input_left.size(1));
                int32_t d_index = pad_index(y + j - disparity, input_left.size(1));
                total_cost += max_val - std::abs(input_left[x_index][y_index] - input_right[x_index][d_index]);
            }
        }
        return total_cost;
    }

};