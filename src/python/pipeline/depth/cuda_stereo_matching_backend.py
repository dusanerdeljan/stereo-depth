import math
from typing import Tuple

import numpy as np
import skimage.measure
from PIL import Image
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from pipeline.depth import StereoMatching


@cuda.jit(device=True, inline=True)
def cuda_2d_grid_coordinates() -> Tuple[int, int]:
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    return x, y


@cuda.jit(device=True)
def compute_cost_function(
        input_left: DeviceNDArray,
        input_right: DeviceNDArray,
        x: int,
        y: int,
        disparity: int,
        patch_radius: int
) -> float:
    total_cost = 0.0
    for xx in range(-patch_radius, patch_radius + 1):
        for yy in range(-patch_radius, patch_radius + 1):
            total_cost += abs(input_left[x, y] - input_right[x, y - disparity])
    return total_cost


@cuda.jit
def wta_cost_volume_construction_kernel(
        input_left: DeviceNDArray,
        input_right: DeviceNDArray,
        output_disparity: DeviceNDArray,
        min_disparity: int,
        max_disparity: int,
        patch_radius: int
) -> None:
    x, y = cuda_2d_grid_coordinates()

    if x >= input_left.shape[0] or y >= input_left.shape[1]:
        return

    best_disparity = min_disparity
    best_cost = 1e38

    for disparity in range(min_disparity, max_disparity + 1):
        cost = compute_cost_function(input_left, input_right, x, y, disparity, patch_radius)
        if cost < best_cost:
            best_cost = cost
            best_disparity = disparity

    output_disparity[x, y] = best_disparity


@cuda.jit
def grayscale_kernel(
        input_image: DeviceNDArray,
        output_image: DeviceNDArray
) -> None:
    x, y = cuda_2d_grid_coordinates()

    if x >= input_image.shape[0] or y >= input_image.shape[1]:
        return

    R = input_image[x, y, 0]
    G = input_image[x, y, 1]
    B = input_image[x, y, 2]

    output_image[x, y] = 0.2989 * R + 0.5870 * G + 0.1140 * B


@cuda.jit
def mean_pool_kernel(
        input_image: DeviceNDArray,
        output_image: DeviceNDArray,
        kernel_size: int
) -> None:
    x, y = cuda_2d_grid_coordinates()

    if x >= output_image.shape[0] or y >= output_image.shape[1]:
        return

    pixel_sum = 0.0
    area = kernel_size ** 2

    for i in range(K):
        for j in range(K):
            pixel_sum += input_image[x * K + i, y * K + j]
    output_image[x, y] = int(pixel_sum / area)


class CudaStereoMatchingBackend(StereoMatching):

    def process(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        H, W, C = left_image.shape
        device_left = cuda.to_device(left_image)
        device_right = cuda.to_device(right_image)
        output_disparity = cuda.device_array(shape=(H, W))

        return output_disparity.copy_to_host()


if __name__ == "__main__":
    left_image = Image.open("../../data/left.png").convert("RGB")
    input_image = np.asarray(left_image)
    H, W, C = input_image.shape

    input_image = cuda.to_device(input_image)

    # GRAYSCALE KERNEL
    grayscale_image = cuda.device_array(shape=(H, W))
    threads = (16, 16)
    blocks = (math.ceil(H / threads[0]), math.ceil(W / threads[1]))
    grayscale_kernel[blocks, threads](input_image, grayscale_image)

    # DOWNSCALE KERNEL
    K = 2
    dH = math.ceil(H / K)
    dW = math.ceil(W / K)
    downscaled_image = cuda.device_array(shape=(dH, dW))
    threads = (16, 16)
    blocks = (math.ceil(dH / threads[0]), math.ceil(dW / threads[1]))
    mean_pool_kernel[blocks, threads](grayscale_image, downscaled_image, K)

    Image.fromarray(downscaled_image.copy_to_host()).show()
