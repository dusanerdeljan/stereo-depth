#include "stereo_matching.hh"

stereo_matching::stereo_matching(const stereo_matching_configuration& configuration)
    : m_config(configuration),
      m_buffer(configuration.height, configuration.width, configuration.min_disparity, configuration.max_disparity, configuration.downscale_factor)
    {}

torch::Tensor stereo_matching::compute_disparity_map(torch::Tensor left_image, torch::Tensor right_image) {
    return m_buffer.output_disparity;
}