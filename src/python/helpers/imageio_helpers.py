import os.path
from typing import Tuple, List

import numpy as np


def rescale_image_to_uint8_range(image: np.ndarray) -> np.ndarray:
    min_val = np.min(image)
    max_val = np.max(image)

    image = image - min_val
    image = image * 255.0 / (max_val - min_val)
    return image


def read_kitti_drive_stereo_pairs(drive_dir: str) -> Tuple[List[str], List[str]]:
    left_image_dir = os.path.join(drive_dir, "image_02", "data")
    right_image_dir = os.path.join(drive_dir, "image_03", "data")

    if not os.path.exists(left_image_dir):
        raise RuntimeError(f"Folder for left images not found: {left_image_dir}.")
    if not os.path.exists(right_image_dir):
        raise RuntimeError(f"Folder for right images not found: {right_image_dir}.")

    left_image_paths = [os.path.join(left_image_dir, image_file) for image_file in os.listdir(left_image_dir)]
    right_image_paths = [os.path.join(right_image_dir, image_file) for image_file in os.listdir(right_image_dir)]
    return left_image_paths, right_image_paths
