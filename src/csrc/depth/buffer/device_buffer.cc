#include "device_buffer.hh"

device_buffer::device_buffer(uint32_t height, uint32_t width, int32_t min_disparity, int32_t max_disparity, uint32_t k)
    : m_options(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)),
      left_grayscaled(torch::empty({height, width}, m_options)),
      right_grayscaled(torch::empty({height, width}, m_options)),
      left_downscaled(torch::empty({(height + k - 1) / k, (width + k - 1) / k}, m_options)),
      right_downscaled(torch::empty({(height + k - 1) / k, (width + k - 1) / k}, m_options)),
      matching_cost_volume(torch::empty({(height + k - 1) / k, (width + k - 1) / k, max_disparity - min_disparity + 1}, m_options)),
      aggregated_cost_volume(torch::empty({(height + k - 1) / k, (width + k - 1) / k}, m_options, max_disparity - min_disparity + 1)),
      downscaled_disparity(torch::empty({(height + k - 1) / k, (width + k - 1) / k}, m_options)),
      output_disparity(torch::empty({height, width}, m_options)) {}