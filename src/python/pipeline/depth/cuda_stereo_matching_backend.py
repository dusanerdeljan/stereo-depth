import math
from typing import Tuple

import numpy as np
import skimage
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
            xx = x + i
            yy = y + j

            left_sum += input_left[xx, yy]
            left_sum_squared += input_left[xx, yy] ** 2

            right_sum += input_right[xx, yy - disparity]
            right_sum_squared += input_right[xx, yy - disparity] ** 2

    left_mean = left_sum / patch_area
    right_mean = right_sum / patch_area

    left_stdev = math.sqrt((left_sum_squared - left_mean**2) / patch_area)
    right_stdev = math.sqrt((right_sum_squared - right_mean**2) / patch_area)

    for i in range(-patch_radius, patch_radius + 1):
        for j in range(-patch_radius, patch_radius + 1):
            xx = x + i
            yy = y + j
            cost += (input_left[xx, yy] - left_mean) * (input_right[xx, yy - disparity] - right_mean)

    total_cost = cost / (patch_area * left_stdev * right_stdev)
    cost_volume[x, y, d] = total_cost


@cuda.jit
def wta_disparity_selection_kernel(
        cost_volume: DeviceNDArray,
        output_disparity: DeviceNDArray,
        min_disparity: int
) -> None:
    x, y = cuda_2d_grid_coordinates()

    if x >= output_disparity.shape[0] or y >= output_disparity.shape[1]:
        return

    best_cost = -1e38
    best_disparity = 0
    for disparity in range(cost_volume.shape[2]):
        if cost_volume[x, y, disparity] > best_cost:
            best_cost = cost_volume[x, y, disparity]
            best_disparity = disparity
    output_disparity[x, y] = best_disparity + min_disparity


@cuda.jit
def multi_block_matching_cost_aggregation_kernel(
        cost_volume: DeviceNDArray,
        block_cost_volume: DeviceNDArray
) -> None:
    x, y, d = cuda_3d_grid_coordinates()

    if x >= cost_volume.shape[0] or y >= cost_volume.shape[1] or d > cost_volume.shape[2]:
        return

    # compute horizontal line block cost (3x21)
    horizontal_cost = 0.0
    for i in range(-1, 2):
        for j in range(-10, 11):
            horizontal_cost += cost_volume[x + i, y + j, d]

    # compute vertical line block cost (21x3)
    vertical_cost = 0.0
    for i in range(-10, 11):
        for j in range(-1, 2):
            vertical_cost += cost_volume[x + i, y + j, d]

    # compute cross block cost (9x9)
    cross_cost = 0.0
    for i in range(-4, 5):
        for j in range(-4, 5):
            cross_cost += cost_volume[x + i, y + j, d]

    total_cost = horizontal_cost * vertical_cost * cross_cost
    block_cost_volume[x, y, d] = total_cost


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
    min_disparity = 75 // K
    max_disparity = 262 // K
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

    # COST VOLUME CONSTRUCTION
    def cost_volume(left, right):
        disparity_range = max_disparity - min_disparity + 1
        ncc_cost_volume = cuda.device_array(shape=(dH, dW, disparity_range))
        threads = (16, 16, 1)
        blocks = (math.ceil(dH / threads[0]), math.ceil(dW / threads[1]), disparity_range)
        ncc_matching_cost_volume_kernel[blocks, threads](left, right, ncc_cost_volume, patch_radius,
                                                         min_disparity, max_disparity)
        return ncc_cost_volume

    matching_cost = cost_volume(downscaled_left, downscaled_right)

    # MULTI BLOCK MATCHING COST AGGREGATION
    def mbm_cost_aggregation(cost):
        disparity_range = max_disparity - min_disparity + 1
        mbm_cost_volume = cuda.device_array(shape=(dH, dW, disparity_range))
        threads = (16, 16, 1)
        blocks = (math.ceil(dH / threads[0]), math.ceil(dW / threads[1]), disparity_range)
        multi_block_matching_cost_aggregation_kernel[blocks, threads](cost, mbm_cost_volume)
        return mbm_cost_volume

    aggregated_cost = mbm_cost_aggregation(matching_cost)

    # WTA DISPARITY SELECTION
    def select_disparity_wta(cost):
        output_disp = cuda.device_array(shape=(dH, dW))
        threads = (16, 16)
        blocks = (math.ceil(dH / threads[0]), math.ceil(dW / threads[1]))
        wta_disparity_selection_kernel[blocks, threads](cost, output_disp, min_disparity)
        return output_disp

    output_disparity = select_disparity_wta(aggregated_cost)
    Image.fromarray(np.round(output_disparity.copy_to_host() * 256).astype(np.uint16)).show()


if __name__ == "__main__":
    main()
