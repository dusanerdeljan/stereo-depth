from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_depth",
    ext_modules=[
        CUDAExtension("cuda_depth", [
            "torch_extension_module.cc",
            "stereo_matching.cc",
            "buffer/device_buffer.cc"
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension
    })