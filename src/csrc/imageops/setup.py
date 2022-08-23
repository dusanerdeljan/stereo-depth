from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_imageops",
    ext_modules=[
        CUDAExtension("cuda_imageops", [
            "torch_extension_module.cc",
            "grayscale_gradient.cc"
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension
    })