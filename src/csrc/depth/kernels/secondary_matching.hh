#pragma once

#include <torch/extension.h>

void secondary_matching_cuda(
    torch::Tensor input_left,
    torch::Tensor input_right,
    torch::Tensor cost_volume,
    torch::Tensor downscaled_disparity,
    int32_t patch_radius,
    int32_t k
);