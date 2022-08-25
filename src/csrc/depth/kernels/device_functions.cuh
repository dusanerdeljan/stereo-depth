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

};