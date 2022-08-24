#include "grayscale_gradient.hh"
#include "rgb_to_grayscale.hh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grayscale_gradient", &image_ops::grayscale_gradient, "Compute grayscale gradient of an image.");
    m.def("rgb_to_grayscale", &image_ops::rgb_to_grayscale, "Convert RGB image to grayscale.");
}