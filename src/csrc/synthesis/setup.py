from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_synthesis",
    ext_modules=[
        CUDAExtension("cuda_synthesis", [
            "torch_extension_module.cc",
            "right_view_synthesis.cc",
            "kernels/rescale_generated_view.cu"
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension
    })