#pragma once

#include <torch/extension.h>

#include "stereo_matching_configuration.hh"
#include "buffer/device_buffer.hh"

class stereo_matching {
public:
    stereo_matching(const stereo_matching_configuration& configuration);

    torch::Tensor compute_disparity_map(torch::Tensor left, torch::Tensor right);

private:
    stereo_matching_configuration m_config;
    device_buffer m_buffer;
};