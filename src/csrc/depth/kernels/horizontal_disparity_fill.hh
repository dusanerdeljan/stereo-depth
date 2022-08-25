#pragma once

#include <torch/extension.h>

void horizontal_disparity_fill_cuda(
    torch::Tensor input_left,
    torch::Tensor upscaled_disparity,
    int32_t k,
    int32_t threshold
);