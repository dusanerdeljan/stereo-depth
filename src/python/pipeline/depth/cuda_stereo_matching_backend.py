from typing import Tuple

import numpy as np
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


class CudaStereoMatchingBackend(StereoMatching):

    def process(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        H, W = left_image.shape
        device_left = cuda.to_device(left_image)
        device_right = cuda.to_device(right_image)
        output_disparity = cuda.device_array(shape=(H, W))

        threads_per_block = (16, 16)
        num_blocks = (
            (H + threads_per_block[0] - 1) // threads_per_block[0],
            (W + threads_per_block[1] - 1) // threads_per_block[1]
        )
        wta_cost_volume_construction_kernel[num_blocks, threads_per_block](
            device_left,
            device_right,
            output_disparity,
            75,
            262,
            7
        )
        return output_disparity.copy_to_host()
