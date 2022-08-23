#include "right_view_synthesis.hh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<right_view_synthesis>(m, "RightViewSynthesis")
        .def(py::init<uint32_t, uint32_t, const std::string&>(),
            py::arg("height"),
            py::arg("width"),
            py::arg("model_path")
        )
        .def("generate_right_view", &right_view_synthesis::generate_right_view);
}