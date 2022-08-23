from typing import Iterator, Tuple, Optional

import numpy as np
from skimage.transform import resize

from helpers.imageio_helpers import load_cv_rgb_image, read_kitti_drive_stereo_pairs
from helpers.paths import python_project_relative_path
from pipeline.camera.camera import Camera


class KittiSingleViewCamera(Camera):

    def __init__(self, drive_dir: str, return_right_view: bool = False, only_one: bool = False):
        self._left_images, self._right_images = read_kitti_drive_stereo_pairs(python_project_relative_path(drive_dir))
        self._left_images.sort()
        self._right_images.sort()
        self._return_right_view = return_right_view
        self._only_one = only_one

    def focal_length(self) -> float:
        return 1.0

    def baseline(self) -> float:
        return 35.0

    def get_image_shape(self) -> Tuple[int, int]:
        return 384, 1280

    def get_disparity_boundaries(self) -> Tuple[int, int]:
        return 1, 64

    def stream_image_pairs(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        for (left_image, right_image) in zip(self._left_images, self._right_images):
            right_view = self._load_view(right_image) if self._return_right_view else None
            yield self._load_view(left_image), right_view
            if self._only_one:
                break

    def _load_view(self, path: str) -> np.ndarray:
        return self._rescale_view(load_cv_rgb_image(path))

    def _rescale_view(self, view: np.ndarray) -> np.ndarray:
        return np.clip(255 * resize(view, self.get_image_shape()) + 0.5, 0, 255).astype(np.uint8)
