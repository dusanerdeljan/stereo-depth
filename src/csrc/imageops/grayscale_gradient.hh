#pragma once

#include <torch/extension.h>

namespace image_ops {
    torch::Tensor grayscale_gradient(torch::Tensor input);
};