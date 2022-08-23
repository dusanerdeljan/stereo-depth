from __future__ import annotations

import os.path
from abc import ABC, abstractmethod

import numpy as np
import skimage.io

from helpers.paths import python_project_relative_path
from helpers.point_cloud_helpers import save_point_cloud_from_depth
from pipeline.camera.camera import Camera


class DisparityMapHook(ABC):

    @abstractmethod
    def process(self, disparity_map: np.ndarray) -> None:
        pass


class DisparityMapCompletionLogger(DisparityMapHook):

    def process(self, disparity_map: np.ndarray) -> None:
        print(f"Computed disparity map: {disparity_map.shape}...")


class DisparityMapSaver(DisparityMapHook):

    def __init__(self, save_dir: str):
        self._save_dir = python_project_relative_path(save_dir)
        self._frame_index = 0

    def process(self, disparity_map: np.ndarray) -> None:
        save_path = os.path.join(self._save_dir, f"disparity_map_{self._frame_index:06d}.png")
        disparity_map = np.round(disparity_map * 256).astype(np.uint16)
        skimage.io.imsave(save_path, disparity_map)
        self._frame_index += 1
        print(f"Saved disparity map: {save_path}...")


class PointCloudSaver(DisparityMapHook):

    def __init__(self,
                 focal_length: float,
                 baseline: float,
                 save_dir: str,
                 invalid_disparity: float):
        self._focal_length = focal_length
        self._baseline = baseline
        self._invalid_disparity = invalid_disparity
        self._save_dir = python_project_relative_path(save_dir)
        self._frame_index = 0

    def process(self, disparity_map: np.ndarray) -> None:
        save_path = os.path.join(self._save_dir, f"point_cloud_{self._frame_index:06d}.ply")
        depth_map = self._disparity_to_depth(disparity_map)
        invalid_disparity_mask = np.logical_not(np.equal(disparity_map, self._invalid_disparity))
        save_point_cloud_from_depth(depth_map, invalid_disparity_mask, save_path)
        self._frame_index += 1
        print(f"Saved point cloud: {save_path}...")

    def _disparity_to_depth(self, disparity_map: np.ndarray) -> np.ndarray:
        return (self._baseline * self._focal_length) / disparity_map

    @staticmethod
    def for_camera(camera: Camera, save_dir: str, invalid_disparity: float) -> PointCloudSaver:
        return PointCloudSaver(
            focal_length=camera.focal_length(),
            baseline=camera.baseline(),
            save_dir=save_dir,
            invalid_disparity=invalid_disparity
        )
