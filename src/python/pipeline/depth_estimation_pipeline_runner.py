from typing import List

from pipeline import DepthEstimationPipeline, DepthEstimationPipelineConfig
from pipeline.camera.camera import Camera
from pipeline.depth_estimation_pipeline import DepthEstimationPipelineContext
from pipeline.depth_estimation_pipeline_hooks import DepthEstimationPipelineHook


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
                                  hooks: List[DepthEstimationPipelineHook] = None) -> None:
    if hooks is None:
        hooks = []

    for frame_index, (left_view, right_view) in enumerate(camera.stream_image_pairs()):
        disparity_map = pipeline.process(left_view, right_view)

        pipeline_context = DepthEstimationPipelineContext(
            disparity_map=disparity_map,
            left_image=left_view,
            right_image=right_view,
            config=pipeline.get_configuration(),
            frame_index=frame_index
        )

        for hook in hooks:
            hook.process(pipeline_context)
