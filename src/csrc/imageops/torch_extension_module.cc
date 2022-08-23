#include "grayscale_gradient.hh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grayscale_gradient", &image_ops::grayscale_gradient, "Compute grayscale gradient of an image.");
}