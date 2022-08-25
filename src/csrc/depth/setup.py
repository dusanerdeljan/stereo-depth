from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_depth",
    ext_modules=[
        CUDAExtension("cuda_depth", [
            "torch_extension_module.cc",
            "stereo_matching.cc",
            "buffer/device_buffer.cc",
            "../imageops/kernels/rgb_to_grayscale.cu",
            "../imageops/kernels/mean_pool.cu",
            "kernels/ncc_matching_cost_volume_construction.cu"
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension
    })