from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_imageops",
    ext_modules=[
        CUDAExtension("cuda_imageops", [
            "torch_extension_module.cc",
            "grayscale_gradient.cc",
            "kernels/rgb_to_grayscale.cu"
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension
    })