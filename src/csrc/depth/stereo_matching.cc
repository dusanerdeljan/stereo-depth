#include "stereo_matching.hh"

#include "../imageops/rgb_to_grayscale.hh"
#include "../imageops/mean_pool.hh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

stereo_matching::stereo_matching(const stereo_matching_configuration& configuration)
    : m_config(configuration),
      m_buffer(configuration.height, configuration.width, configuration.min_disparity, configuration.max_disparity, configuration.downscale_factor)
    {}

torch::Tensor stereo_matching::compute_disparity_map(torch::Tensor left_image, torch::Tensor right_image) {
    CHECK_INPUT(left_image);
    CHECK_INPUT(right_image);

    image_ops::rgb_to_grayscale_inplace(left_image, m_buffer.left_grayscaled);
    image_ops::rgb_to_grayscale_inplace(right_image, m_buffer.right_grayscaled);

    image_ops::mean_pool_inplace(m_buffer.left_grayscaled, m_buffer.left_downscaled, m_config.downscale_factor);
    image_ops::mean_pool_inplace(m_buffer.right_grayscaled, m_buffer.left_downscaled, m_config.downscale_factor);

    return m_buffer.output_disparity;
}