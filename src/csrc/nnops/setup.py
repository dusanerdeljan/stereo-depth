from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_nn_ops",
    ext_modules=[
        CUDAExtension("cuda_nn_ops", [
            "torch_extension_module.cc",
            "kernels/disparity_shift_stack.cu"
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension
    })