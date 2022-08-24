import math
from typing import Tuple

import numpy as np
from PIL import Image
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from pipeline.depth import StereoMatching


@cuda.jit(device=True, inline=True)
def cuda_2d_grid_coordinates() -> Tuple[int, int]:
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    return x, y


@cuda.jit(device=True, inline=True)
def cuda_3d_grid_coordinates() -> Tuple[int, int, int]:
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    z = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    return x, y, z


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

    for i in range(kernel_size):
        for j in range(kernel_size):
            pixel_sum += input_image[x * kernel_size + i, y * kernel_size + j]
    output_image[x, y] = int(pixel_sum / area)


@cuda.jit
def ncc_matching_cost_volume_kernel(
        input_left: DeviceNDArray,
        input_right: DeviceNDArray,
        cost_volume: DeviceNDArray,
        patch_radius: int,
        min_disparity: int,
        max_disparity: int
) -> None:
    x, y, d = cuda_3d_grid_coordinates()
    disparity = min_disparity + d

    if x >= input_left.shape[0] or y >= input_right.shape[1] or disparity > max_disparity:
        return

    cost = 0.0
    left_sum = 0.0
    right_sum = 0.0
    left_sum_squared = 0.0
    right_sum_squared = 0.0
    patch_area = (2 * patch_radius + 1) ** 2
    for i in range(-patch_radius, patch_radius + 1):
        for j in range(-patch_radius, patch_radius + 1):
            left_sum += input_left[i, j]
            left_sum_squared += input_left[i, j] ** 2

            right_sum += input_right[i, j - disparity]
            right_sum_squared += input_right[i, j - disparity] ** 2

    left_mean = left_sum / patch_area
    right_mean = right_sum / patch_area

    left_stdev = math.sqrt((left_sum_squared - left_mean**2) / patch_area)
    right_stdev = math.sqrt((right_sum_squared - right_mean**2) / patch_area)

    for i in range(-patch_radius, patch_radius + 1):
        for j in range(-patch_radius, patch_radius + 1):
            cost += (input_left[i, j] - left_mean) * (input_right[i, j - disparity] - right_mean)

    cost_volume[x, y, d] = cost / (patch_area * left_stdev * right_stdev)


class CudaStereoMatchingBackend(StereoMatching):

    def process(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        H, W, C = left_image.shape
        device_left = cuda.to_device(left_image)
        device_right = cuda.to_device(right_image)
        output_disparity = cuda.device_array(shape=(H, W))

        return output_disparity.copy_to_host()


def main():
    left_image = np.asarray(Image.open("../../data/left.png").convert("RGB"))
    right_image = np.asarray(Image.open("../../data/right.png").convert("RGB"))
    H, W, C = left_image.shape
    K = 2
    dH = math.ceil(H / K)
    dW = math.ceil(W / K)
    min_disparity = 75
    max_disparity = 262
    patch_radius = 1

    left_image = cuda.to_device(left_image)
    right_image = cuda.to_device(right_image)

    # GRAYSCALE KERNEL
    def grayscale(input_image):
        grayscale_image = cuda.device_array(shape=(H, W))
        threads = (16, 16)
        blocks = (math.ceil(H / threads[0]), math.ceil(W / threads[1]))
        grayscale_kernel[blocks, threads](input_image, grayscale_image)
        return grayscale_image

    grayscale_left = grayscale(left_image)
    grayscale_right = grayscale(right_image)

    # DOWNSCALE KERNEL
    def downscale(grayscale_image):
        downscaled_image = cuda.device_array(shape=(dH, dW))
        threads = (16, 16)
        blocks = (math.ceil(dH / threads[0]), math.ceil(dW / threads[1]))
        mean_pool_kernel[blocks, threads](grayscale_image, downscaled_image, K)
        return downscaled_image

    downscaled_left = downscale(grayscale_left)
    downscaled_right = downscale(grayscale_right)

    def cost_volume(left, right):
        disparity_range = max_disparity - min_disparity + 1
        ncc_cost_volume = cuda.device_array(shape=(dH, dW, disparity_range))
        threads = (16, 16, 1)
        blocks = (math.ceil(dH / threads[0]), math.ceil(dW / threads[1]), disparity_range)
        ncc_matching_cost_volume_kernel[blocks, threads](left, right, ncc_cost_volume, patch_radius,
                                                         min_disparity, max_disparity)
        return ncc_cost_volume

    matching_cost = cost_volume(downscaled_left, downscaled_right)
    print(matching_cost.shape)

    Image.fromarray(downscaled_right.copy_to_host()).show()


if __name__ == "__main__":
    main()
