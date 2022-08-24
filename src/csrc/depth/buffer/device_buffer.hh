#pragma once

#include <torch/extension.h>

class device_buffer {
private:
    torch::TensorOptions m_options;

public:
    device_buffer(uint32_t height, uint32_t width, int32_t min_disparity, int32_t max_disparity, uint32_t k);

    torch::Tensor left_grayscaled;
    torch::Tensor right_grayscaled;
    torch::Tensor left_downscaled;
    torch::Tensor right_downscaled;
    torch::Tensor matching_cost_volume;
    torch::Tensor aggregated_cost_volume;
    torch::Tensor downscaled_disparity;
    torch::Tensor output_disparity;
};