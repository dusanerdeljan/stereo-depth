#pragma once

#include <torch/extension.h>

namespace image_ops {
    torch::Tensor rgb_to_grayscale(torch::Tensor input);

    void rgb_to_grayscale_inplace(torch::Tensor input, torch::Tensor output);
};