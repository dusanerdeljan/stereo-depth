#pragma once

#include <torch/extension.h>

#include "stereo_matching_configuration.hh"
#include "buffer/device_buffer.hh"

class stereo_matching {
public:
    stereo_matching(const stereo_matching_configuration& configuration);

    torch::Tensor compute_disparity_map(torch::Tensor left_image, torch::Tensor right_image);

private:
    void grayscale(torch::Tensor left_image, torch::Tensor right_image);

    void downscale();

    void ncc_matching_cost_volume_construction();

    void multi_block_matching_cost_aggregation();

    void wta_disparity_selection();

    void secondary_matching();

    void upscale_disparity_vertical_fill();

    void horizontal_disparity_fill();

    stereo_matching_configuration m_config;
    device_buffer m_buffer;
};