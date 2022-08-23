#pragma once

#include <torch/extension.h>

namespace nn_ops {
    torch::Tensor disparity_shift_stack(
        torch::Tensor left_image,
        int32_t min_disparity,
        int32_t max_disparity
    );
};