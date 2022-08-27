from typing import Iterable

from joblib import Parallel, delayed, cpu_count

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
                                  hooks: Iterable[DepthEstimationPipelineHook] = None) -> None:
    if hooks is None:
        hooks = []

    n_parallel_jobs = min(len(hooks), cpu_count() - 1, 1)
    with Parallel(n_jobs=n_parallel_jobs) as parallel_thread_pool:
        for frame_index, (left_view, right_view) in enumerate(camera.stream_image_pairs()):
            disparity_map, (left_image, right_image) = pipeline.process(left_view, right_view)

            pipeline_context = DepthEstimationPipelineContext(
                disparity_map=disparity_map,
                left_image=left_image,
                right_image=right_image,
                config=pipeline.get_configuration(),
                frame_index=frame_index
            )

            parallel_thread_pool(
                delayed(DepthEstimationPipelineHook.invoke_in_context)(hook, pipeline_context) for hook in hooks
            )
