import os.path
from typing import List, Tuple

import torch.utils.data

from helpers.imageio_helpers import to_resized_torch_image, load_cv_rgb_image, read_kitti_drive_stereo_pairs


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

        self._load_stereo_images(data_path, date, drives)

    def __len__(self) -> int:
        return len(self._left_images)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        if index >= len(self._left_images):
            raise RuntimeError("Index out of range.")

        left_image = load_cv_rgb_image(self._left_images[index])
        right_image = load_cv_rgb_image(self._right_images[index])

        left_full_resolution = to_resized_torch_image(left_image, self._full_resolution)
        left_downscaled = to_resized_torch_image(left_image, self._downscaled_resolution)
        right_full_resolution = to_resized_torch_image(right_image, self._full_resolution)

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


def test_kitti_dataset():
    full_resolution = (375, 1242)
    downscale_factor = 4
    downscaled_resolution = (full_resolution[0] // downscale_factor, full_resolution[1] // downscale_factor)
    kitti_dataset = KittiStereoDataset(
        data_path="../../data/train",
        date="2011_09_26",
        drives=["0011", "0019", "0022", "0052", "0059", "0084", "0091", "0093", "0095", "0096"],
        full_resolution=full_resolution,
        downscaled_resolution=downscaled_resolution
    )
    print(len(kitti_dataset))
    left, left_downscaled, right_gt = kitti_dataset[1234]
    print(left.shape)
    print(left_downscaled.shape)
    print(right_gt.shape)


if __name__ == "__main__":
    test_kitti_dataset()
