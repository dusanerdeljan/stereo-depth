#include "stereo_matching.hh"
#include "stereo_matching_configuration.hh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<stereo_matching_configuration>(m, "StereoMatchingConfiguration")
        .def(py::init<uint32_t, uint32_t, uint32_t, int32_t, int32_t, uint32_t, uint32_t, uint32_t, int32_t, int32_t, int32_t>(),
            py::arg("height") = 1080,
            py::arg("width") = 1980,
            py::arg("downscale_factor") = 2,
            py::arg("min_disparity") = 75,
            py::arg("max_disparity") = 262,
            py::arg("ncc_patch_radius") = 1,
            py::arg("sad_patch_radius") = 5,
            py::arg("threshold") = 5,
            py::arg("small_mbm_radius") = 1,
            py::arg("mid_mbm_radius") = 4,
            py::arg("large_mbm_radius") = 10
        );

    py::class_<stereo_matching>(m, "StereoMatching")
        .def(py::init<const stereo_matching_configuration&>(),
            py::arg("configuration") = stereo_matching_configuration{}
        )
        .def("compute_disparity_map", &stereo_matching::compute_disparity_map);
}