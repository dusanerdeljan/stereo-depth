#pragma once

#include <torch/extension.h>

void ncc_matching_cost_volume_construction_cuda(
    torch::Tensor input_left,
    torch::Tensor input_right,
    torch::Tensor cost_volume,
    int32_t patch_radius,
    int32_t min_disparity,
    int32_t max_disparity
);