from __future__ import annotations

import os.path
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable

import torch
import torchvision.utils

from helpers.imageio_helpers import save_image_grid, prepare_image_grid
from helpers.paths import timestamp_folder_name
from helpers.point_cloud_helpers import save_point_cloud_from_depth
from pipeline.camera.camera import Camera
from pipeline.depth_estimation_pipeline import DepthEstimationPipelineContext


class DepthEstimationPipelineHook(ABC):

    @abstractmethod
    def process(self, context: DepthEstimationPipelineContext) -> None:
        pass

    def on_pipeline_start(self):
        pass

    def on_pipeline_end(self):
        pass

    @staticmethod
    def invoke_in_context(hook: DepthEstimationPipelineHook, context: DepthEstimationPipelineContext) -> None:
        hook.process(context)


class LambdaHook(DepthEstimationPipelineHook):

    def __init__(self, func: Callable[[DepthEstimationPipelineContext], None]):
        self._func = func

    def process(self, context: DepthEstimationPipelineContext) -> None:
        self._func(context)


class DisparityMapCompletionLogger(DepthEstimationPipelineHook):

    def process(self, context: DepthEstimationPipelineContext) -> None:
        print(f"Computed disparity map: {context.disparity_map.shape}...")


class DisparityMapSaver(DepthEstimationPipelineHook):

    def __init__(self, save_dir: str):
        self._save_dir = os.path.join(save_dir, timestamp_folder_name())
        os.makedirs(self._save_dir, exist_ok=True)

    def process(self, context: DepthEstimationPipelineContext) -> None:
        save_path = os.path.join(self._save_dir, f"disparity_map_{context.frame_index:06d}.png")
        save_image_grid(context.disparity_map, save_path)


class ContextFrameSaver(DepthEstimationPipelineHook):

    def __init__(self, save_dir: str):
        self._save_dir = os.path.join(save_dir, timestamp_folder_name())
        os.makedirs(self._save_dir, exist_ok=True)

    def process(self, context: DepthEstimationPipelineContext) -> None:
        save_path = os.path.join(self._save_dir, f"context_frame_{context.frame_index:06d}.png")
        save_image_grid([context.left_image, context.right_image, context.disparity_map], save_path)


class PointCloudSaver(DepthEstimationPipelineHook):

    def __init__(self,
                 focal_length: float,
                 baseline: float,
                 save_dir: str,
                 invalid_disparity: float):
        self._focal_length = focal_length
        self._baseline = baseline
        self._invalid_disparity = invalid_disparity
        self._save_dir = os.path.join(save_dir, timestamp_folder_name())

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


class ContextVideoSaver(DepthEstimationPipelineHook):

    def __init__(self, save_path: str, fps: int):
        self._fps = fps
        self._save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._frames = OrderedDict()

    def on_pipeline_end(self):
        torchvision.io.write_video(
            filename=self._save_path,
            video_array=torch.stack([self._frames[frame_idx] for frame_idx in self._frames.keys()], dim=0),
            fps=self._fps
        )

    def process(self, context: DepthEstimationPipelineContext) -> None:
        frame = ContextVideoSaver._extract_frame_from_context(context)
        self._frames[context.frame_index] = frame

    @staticmethod
    def _extract_frame_from_context(context: DepthEstimationPipelineContext) -> torch.Tensor:
        images = prepare_image_grid([
            context.left_image,
            context.right_image,
            context.disparity_map
        ])
        grid = torchvision.utils.make_grid(images, padding=10, pad_value=255)
        return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8)
