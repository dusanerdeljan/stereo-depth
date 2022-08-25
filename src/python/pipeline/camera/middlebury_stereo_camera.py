import os.path
from dataclasses import dataclass
from typing import Tuple, Iterator, Optional

import numpy as np
import torch
import torchvision.io

from helpers.paths import python_project_relative_path
from pipeline.camera.camera import Camera


@dataclass
class MiddleBuryStereoCameraCalibration:
    cam0: np.ndarray
    cam1: np.ndarray
    doffs: float
    baseline: float
    width: int
    height: int
    ndisp: int
    vmin: int
    vmax: int

    @property
    def fx(self) -> float:
        return self.cam0[0, 0]

    @property
    def fy(self) -> float:
        return self.cam0[1, 1]

    @property
    def cx(self) -> float:
        return self.cam0[0, 2]

    @property
    def cy(self) -> float:
        return self.cam0[1, 2]

    def get_focal_length(self) -> Tuple[float, float]:
        return self.fx, self.fy

    def get_principal_point(self) -> Tuple[float, float]:
        return self.cx, self.cy


class MiddleBuryStereoCamera(Camera):

    def __init__(self, middlebury_dir: str) -> None:
        middlebury_dir = python_project_relative_path(middlebury_dir)
        if not os.path.exists(middlebury_dir):
            raise RuntimeError(f"Directory '{middlebury_dir}' not found.")

        left_image_path = os.path.join(middlebury_dir, "left.png")
        right_image_path = os.path.join(middlebury_dir, "right.png")
        calibration_file_path = os.path.join(middlebury_dir, "calib.txt")

        self._left_image = torchvision.io.read_image(left_image_path)
        self._right_image = torchvision.io.read_image(right_image_path)
        self._calibration = MiddleBuryStereoCamera._load_calibration_file(calibration_file_path)

    def focal_length(self) -> float:
        return self._calibration.fx

    def baseline(self) -> float:
        return self._calibration.baseline

    def get_image_shape(self) -> Tuple[int, int]:
        return self._calibration.height, self._calibration.width

    def get_disparity_boundaries(self) -> Tuple[int, int]:
        return self._calibration.vmin, self._calibration.vmax

    def stream_image_pairs(self) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        yield self._left_image, self._right_image

    @staticmethod
    def _load_camera_intrinsics(intrinsics: str) -> np.ndarray:
        return np.array(
            [[float(x.strip()) for x in arr.strip().split(" ")]
             for arr in intrinsics.replace("[", "").replace("]", "").split(";")]
        )

    @staticmethod
    def _load_calibration_file(calibration_file_path: str) -> MiddleBuryStereoCameraCalibration:
        calibration_data = {}
        calibration_data_parsers = {
            "cam0": MiddleBuryStereoCamera._load_camera_intrinsics,
            "cam1": MiddleBuryStereoCamera._load_camera_intrinsics,
            "doffs": float,
            "baseline": float,
            "width": int,
            "height": int,
            "ndisp": int,
            "vmin": int,
            "vmax": int
        }

        with open(calibration_file_path, "r") as calibration_file:
            for line in calibration_file:
                key, value = line.split("=")
                calibration_data[key] = calibration_data_parsers[key](value)
        return MiddleBuryStereoCameraCalibration(**calibration_data)
