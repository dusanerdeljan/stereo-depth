import os.path
from typing import Tuple, List, Union

import torch
import torchvision.utils


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    return image / 256


def ensure_chw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image
    return torch.tile(image.unsqueeze(0), (3, 1, 1))


def save_image_grid(images: Union[torch.Tensor, List[torch.Tensor]],
                    file_path: str,
                    padding: int = 10,
                    pad_value: int = 255) -> None:
    if not isinstance(images, list):
        images = [images]
    images = [ensure_chw(normalize_image(image)).cpu() for image in images]
    torchvision.utils.save_image(images, file_path, padding=padding, pad_value=pad_value)


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
