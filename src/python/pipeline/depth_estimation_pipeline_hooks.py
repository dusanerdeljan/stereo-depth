from __future__ import annotations

import os.path
from abc import ABC, abstractmethod

import torch
import torchvision.utils

from helpers.paths import python_project_relative_path
from helpers.point_cloud_helpers import save_point_cloud_from_depth
from pipeline.camera.camera import Camera
from pipeline.depth_estimation_pipeline import DepthEstimationPipelineContext


class DepthEstimationPipelineHook(ABC):

    @abstractmethod
    def process(self, context: DepthEstimationPipelineContext) -> None:
        pass


class DisparityMapCompletionLogger(DepthEstimationPipelineHook):

    def process(self, context: DepthEstimationPipelineContext) -> None:
        print(f"Computed disparity map: {context.disparity_map.shape}...")


class DisparityMapSaver(DepthEstimationPipelineHook):

    def __init__(self, save_dir: str):
        self._save_dir = python_project_relative_path(save_dir)

    def process(self, context: DepthEstimationPipelineContext) -> None:
        save_path = os.path.join(self._save_dir, f"disparity_map_{context.frame_index:06d}.png")
        torchvision.utils.save_image(context.disparity_map / 256, save_path)
        print(f"Saved disparity map: {save_path}...")


class PointCloudSaver(DepthEstimationPipelineHook):

    def __init__(self,
                 focal_length: float,
                 baseline: float,
                 save_dir: str,
                 invalid_disparity: float):
        self._focal_length = focal_length
        self._baseline = baseline
        self._invalid_disparity = invalid_disparity
        self._save_dir = python_project_relative_path(save_dir)

    def process(self, context: DepthEstimationPipelineContext) -> None:
        save_path = os.path.join(self._save_dir, f"point_cloud_{context.frame_index:06d}.ply")
        depth_map = self._disparity_to_depth(context.disparity_map)
        invalid_disparity_mask = torch.logical_not(torch.eq(context.disparity_map, self._invalid_disparity))
        save_point_cloud_from_depth(depth_map.cpu(), invalid_disparity_mask.cpu(), save_path)
        print(f"Saved point cloud: {save_path}...")

    def _disparity_to_depth(self, disparity_map: torch.Tensor) -> torch.Tensor:
        return (self._baseline * self._focal_length) / disparity_map

    @staticmethod
    def for_camera(camera: Camera, save_dir: str, invalid_disparity: float) -> PointCloudSaver:
        return PointCloudSaver(
            focal_length=camera.focal_length(),
            baseline=camera.baseline(),
            save_dir=save_dir,
            invalid_disparity=invalid_disparity
        )