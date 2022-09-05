import os.path
from typing import Iterator, Tuple, Optional

import torch
import torchvision.io
import torchvision.transforms as T

from helpers.imageio_helpers import read_kitti_drive_stereo_pairs
from helpers.paths import python_project_relative_path
from helpers.velodyne_points_helpers import generate_depth_map
from pipeline.camera.camera import Camera


class KittiSingleViewCamera(Camera):

    def __init__(self, drive_dir: str, return_right_view: bool = False, only_one: bool = False):
        self._calib_dir = os.path.dirname(python_project_relative_path(drive_dir))
        self._left_images, self._right_images = read_kitti_drive_stereo_pairs(python_project_relative_path(drive_dir))
        self._left_images.sort()
        self._right_images.sort()
        self._return_right_view = return_right_view
        self._only_one = only_one
        self._pil_to_tensor = T.PILToTensor()
        self._resize = T.Resize(size=self.get_image_shape())

    def focal_length(self) -> float:
        return 721.0

    def baseline(self) -> float:
        return 0.54

    def get_image_shape(self) -> Tuple[int, int]:
        return 384, 1280

    def get_disparity_boundaries(self) -> Tuple[int, int]:
        return 1, 64

    def stream_image_pairs(self) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        for (left_image, right_image) in zip(self._left_images, self._right_images):
            right_view = self._load_view(right_image) if self._return_right_view else None
            yield self._load_view(left_image), right_view
            if self._only_one:
                break

    def stream_image_pairs_with_gt_disparity(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for (left_image, right_image) in zip(self._left_images, self._right_images):
            left_view = self._load_view(left_image)
            right_view = self._load_view(right_image)
            velodyne_disparity = self._load_velodyne_gt_disparity_map(left_image)
            yield left_view, right_view, velodyne_disparity
            if self._only_one:
                break

    def _load_view(self, path: str) -> torch.Tensor:
        return self._resize(torchvision.io.read_image(path))

    def _load_velodyne_gt_disparity_map(self, left_image_path: str) -> torch.Tensor:
        velodyne_depth = torch.from_numpy(generate_depth_map(
            calib_dir=self._calib_dir,
            velo_filename=KittiSingleViewCamera._left_image_path_to_velodyne_path(left_image_path)
        ))
        velodyne_disparity = self._depth_to_disparity(velodyne_depth)
        velodyne_disparity[velodyne_disparity == torch.inf] = 0
        return self._resize(velodyne_disparity.unsqueeze(0)).squeeze(0)

    def _depth_to_disparity(self, depth: torch.Tensor) -> torch.Tensor:
        return self.baseline() * self.focal_length() / depth

    @staticmethod
    def _left_image_path_to_velodyne_path(path: str) -> str:
        return path.replace("image_02", "velodyne_points").replace(".png", ".bin")

