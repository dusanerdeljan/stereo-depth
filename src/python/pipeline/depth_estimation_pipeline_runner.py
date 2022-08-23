from typing import List

import torch.cuda

from pipeline import DepthEstimationPipeline, DepthEstimationPipelineConfig
from pipeline.camera.camera import Camera
from pipeline.disparity_map_hooks import DisparityMapHook


def extract_config_from_camera(camera: Camera) -> DepthEstimationPipelineConfig:
    min_disparity, max_disparity = camera.get_disparity_boundaries()
    config = DepthEstimationPipelineConfig(
        image_shape=camera.get_image_shape(),
        min_disparity=min_disparity,
        max_disparity=max_disparity
    )
    return config


def run_depth_estimation_pipeline(camera: Camera,
                                  pipeline: DepthEstimationPipeline,
                                  hooks: List[DisparityMapHook] = None) -> None:
    torch.cuda.empty_cache()
    if hooks is None:
        hooks = []

    for left_view, right_view in camera.stream_image_pairs():
        disparity_map = pipeline.process(left_view, right_view)
        for hook in hooks:
            hook.process(disparity_map)
