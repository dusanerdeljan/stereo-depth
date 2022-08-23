#include "kernels/disparity_shift_stack.hh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("disparity_shift_stack", &nn_ops::disparity_shift_stack, "Compute shifted view disparity stack.");
}