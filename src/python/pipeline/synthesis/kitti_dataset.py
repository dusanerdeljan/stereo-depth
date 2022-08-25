import os.path
from typing import List, Tuple

import torch.utils.data
import torchvision.io
import torchvision.transforms as T

from helpers.imageio_helpers import read_kitti_drive_stereo_pairs


class KittiStereoDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_path: str,
                 date: str,
                 drives: List[str],
                 full_resolution: Tuple[int, int],
                 downscaled_resolution: Tuple[int, int]):
        self._left_images = []
        self._right_images = []

        self._full_resolution = full_resolution
        self._downscaled_resolution = downscaled_resolution

        self._resize_to_full_resolution = T.Resize(size=self._full_resolution)
        self._resize_to_downscaled_resolution = T.Resize(size=self._downscaled_resolution)

        self._load_stereo_images(data_path, date, drives)

    def __len__(self) -> int:
        return len(self._left_images)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        if index >= len(self._left_images):
            raise RuntimeError("Index out of range.")

        left_image = torchvision.io.read_image(self._left_images[index])
        right_image = torchvision.io.read_image(self._right_images[index])

        left_full_resolution = self._resize_to_full_resolution(left_image)
        left_downscaled = self._resize_to_downscaled_resolution(right_image)
        right_full_resolution = self._resize_to_full_resolution(right_image)

        return left_full_resolution, left_downscaled, right_full_resolution

    def _load_stereo_images(self, data_path: str, date: str, drives: List[str]) -> None:
        if not os.path.exists(data_path):
            raise RuntimeError("Dataset root path not found.")

        drive_dataset_path = os.path.join(data_path, date)
        if not os.path.exists(drive_dataset_path):
            raise RuntimeError(f"Dataset for date {date} not found.")

        for drive in drives:
            self._process_single_drive(date, drive, drive_dataset_path)

        self._left_images.sort()
        self._right_images.sort()

    def _process_single_drive(self, date: str, drive: str, drive_dataset_path: str) -> None:
        single_drive_path = os.path.join(drive_dataset_path, f"{date}_drive_{drive}_sync")
        if not os.path.exists(single_drive_path):
            raise RuntimeError(f"Drive path not found for drive: {drive}.")

        left_image_paths, right_image_paths = read_kitti_drive_stereo_pairs(single_drive_path)

        self._left_images.extend(left_image_paths)
        self._right_images.extend(right_image_paths)
