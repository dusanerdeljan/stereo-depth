#include "stereo_matching.hh"

#include "../imageops/rgb_to_grayscale.hh"
#include "../imageops/mean_pool.hh"

#include "kernels/ncc_matching_cost_volume_construction.hh"
#include "kernels/multi_block_matching_cost_aggregation.hh"
#include "kernels/wta_disparity_selection.hh"

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
    image_ops::mean_pool_inplace(m_buffer.right_grayscaled, m_buffer.right_downscaled, m_config.downscale_factor);

    ncc_matching_cost_volume_construction();

    multi_block_matching_cost_aggregation();

    wta_disparity_selection();

    return m_buffer.downscaled_disparity;
}

void stereo_matching::ncc_matching_cost_volume_construction() {
    std::cout << "calling ncc_matching_cost_volume_construction_cuda" << std::endl;
    ncc_matching_cost_volume_construction_cuda(
        m_buffer.left_downscaled,
        m_buffer.right_downscaled,
        m_buffer.matching_cost_volume,
        m_config.ncc_patch_radius,
        m_config.min_disparity / m_config.downscale_factor,
        m_config.max_disparity / m_config.downscale_factor
    );
}

void stereo_matching::multi_block_matching_cost_aggregation() {
    std::cout << "calling multi_block_matching_cost_aggregation" << std::endl;
    multi_block_matching_cost_aggregation_cuda(
        m_buffer.matching_cost_volume,
        m_buffer.aggregated_cost_volume,
        m_config.min_disparity / m_config.downscale_factor,
        m_config.max_disparity / m_config.downscale_factor,
        1,
        4,
        10
    );
}

void stereo_matching::wta_disparity_selection() {
    std::cout << "calling wta_disparity_selection_cuda" << std::endl;
    wta_disparity_selection_cuda(
        m_buffer.aggregated_cost_volume,
        m_buffer.downscaled_disparity,
        m_config.min_disparity / m_config.downscale_factor
    );
}