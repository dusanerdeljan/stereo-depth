#pragma once

#include <torch/extension.h>

void multi_block_matching_cost_aggregation_cuda(
    torch::Tensor cost_volume,
    torch::Tensor block_cost_volume,
    int32_t min_disparity,
    int32_t max_disparity,
    int32_t small_radius,
    int32_t mid_radius,
    int32_t large_radius
);