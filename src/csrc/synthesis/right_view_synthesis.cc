#include "right_view_synthesis.hh"
#include "kernels/rescale_generated_view.hh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

right_view_synthesis::right_view_synthesis(uint32_t height, uint32_t width, const std::string& model_path)
    : m_model(torch::jit::load(model_path)),
      m_output_buffer(torch::empty({3, height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))) {}

torch::Tensor right_view_synthesis::generate_right_view(torch::Tensor left_full, torch::Tensor left_downscaled) {
    CHECK_INPUT(left_full);
    CHECK_INPUT(left_downscaled);
    std::vector<torch::jit::IValue> inputs {
        left_full.unsqueeze(0), left_downscaled.unsqueeze(0)
    };
    torch::Tensor generated_right_view = m_model.forward(inputs).toTensor();
    rescale_generated_view_cuda(generated_right_view, m_output_buffer);
    return m_output_buffer;
}