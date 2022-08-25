#pragma once

#include <torch/extension.h>

void upscale_disparity_vertical_fill_cuda(
    torch::Tensor input_left,
    torch::Tensor downscaled_disparity,
    torch::Tensor upscaled_disparity,
    int32_t k,
    int32_t threshold
);