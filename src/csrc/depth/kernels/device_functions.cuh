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
    __device__ __forceinline__ scalar_t quadratic_function_peak(
        scalar_t x1,
        scalar_t y1,
        scalar_t x2,
        scalar_t y2,
        scalar_t x3,
        scalar_t y3
    ) {
        scalar_t denominator = (x1 - x2) * (x2 - x3) * (x1 - x3);
        scalar_t min_value;
        if (y1 > y2) {
            min_value = (y1 > y3) ? x1 : x3;
        } else {
            min_value = (y2 > y3) ? x2 : x3;
        }
        if (denominator != 0) {
            scalar_t a = x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2);
            scalar_t b = x1*x1 * (y2 - y3) + x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1);
            if (a < 0) {
                min_value = -b / (2 * a);
            }
        }
        return min_value;
    }

    template<typename scalar_t>
    __device__ __forceinline__ bool have_same_sign(scalar_t a, scalar_t b) {
        return (a * b) > 0;
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