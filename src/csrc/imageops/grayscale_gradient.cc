#include "grayscale_gradient.hh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace image_ops {
    torch::Tensor grayscale_gradient(torch::Tensor input) {
        CHECK_INPUT(input);

        static torch::Tensor kernel = torch::tensor({{1, 0, -1},
                                                     {2, 0, -2},
                                                     {1, 0, -1}}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto x = input.unsqueeze(0);
        auto G_x = torch::conv2d(x, kernel.view({1, 1, 3, 3}));
        auto G_y = torch::conv2d(x, kernel.t().view({1, 1, 3, 3}));
        auto G = torch::sqrt(torch::pow(G_x, 2) + torch::pow(G_y, 2));
        return G.squeeze(0);
    }
};