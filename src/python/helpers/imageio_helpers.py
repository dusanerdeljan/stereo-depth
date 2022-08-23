import os.path
from typing import Tuple, List

import cv2
import numpy as np
import torch
from skimage.transform import resize


def save_debug_image(image: np.ndarray, file_path: str) -> None:
    image = rescale_image_to_uint8_range(image)
    cv2.imwrite(file_path, image)


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


def chw_ordering(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute([-1, 0, 1]).contiguous()


def resize_image(image: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
    return resize(image, resolution)


def to_resized_torch_image(image: np.ndarray, resolution: Tuple[int, int]) -> torch.Tensor:
    return chw_ordering(torch.from_numpy(resize_image(image, resolution)))


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def load_cv_rgb_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image {image_path} not found.")
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def save_cv_rgb_image(image_path: str, image: np.ndarray) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)
