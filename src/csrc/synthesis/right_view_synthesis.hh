#pragma once

#include <torch/extension.h>
#include <torch/script.h>

#include <cstdint>
#include <string>

class right_view_synthesis {
public:
    right_view_synthesis(uint32_t height, uint32_t width, const std::string& model_path);

    torch::Tensor generate_right_view(torch::Tensor left_full, torch::Tensor left_downscaled);

private:
    torch::jit::script::Module m_model;
    torch::Tensor m_output_buffer;
};