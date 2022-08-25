#pragma once

#include <torch/extension.h>

void wta_disparity_selection_cuda(
    torch::Tensor cost_volume,
    torch::Tensor downscaled_disparity,
    int32_t min_disparity
);