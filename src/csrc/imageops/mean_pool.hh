#pragma once

#include <torch/extension.h>

namespace image_ops {
    torch::Tensor mean_pool(torch::Tensor input, uint32_t kernel_size);

    void mean_pool_inplace(torch::Tensor input, torch::Tensor output, uint32_t kernel_size);
};