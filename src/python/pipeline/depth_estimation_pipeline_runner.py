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


def validate_pipeline_config_wrt_camera(config: DepthEstimationPipelineConfig, camera: Camera) -> None:
    if camera.get_image_shape() != config.image_shape:
        raise RuntimeError(f"Incompatible image shapes between pipeline configuration and camera."
                           f"Pipeline expects: {config.image_shape} but camera provides: {camera.get_image_shape()}.")


def run_depth_estimation_pipeline(camera: Camera,
                                  pipeline: DepthEstimationPipeline,
                                  hooks: Iterable[DepthEstimationPipelineHook] = None) -> None:
    if hooks is None:
        hooks = []

    validate_pipeline_config_wrt_camera(pipeline.get_configuration(), camera)

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


def run_depth_estimation_pipeline_evaluation(camera: Camera,
                                             pipeline: DepthEstimationPipeline):
    validate_pipeline_config_wrt_camera(pipeline.get_configuration(), camera)

    for frame_index, (left_view, right_view, gt_disparity) in enumerate(camera.stream_image_pairs_with_gt_disparity()):
        disparity_map, (left_view, right_view) = pipeline.process(left_view, right_view)
        print(gt_disparity.shape)
