#pragma once

#include <torch/extension.h>

#include "stereo_matching_configuration.hh"
#include "buffer/device_buffer.hh"

class stereo_matching {
public:
    stereo_matching(const stereo_matching_configuration& configuration);

    torch::Tensor compute_disparity_map(torch::Tensor left, torch::Tensor right);

private:
    void ncc_matching_cost_volume_construction();

    void multi_block_matching_cost_aggregation();

    void wta_disparity_selection();

    stereo_matching_configuration m_config;
    device_buffer m_buffer;
};