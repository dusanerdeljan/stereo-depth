#pragma once

#include <torch/extension.h>

void rescale_generated_view_cuda(
    torch::Tensor input,
    torch::Tensor output
);