from typing import Iterable, Literal, Dict, List

from joblib import Parallel, delayed, cpu_count

from pipeline import DepthEstimationPipeline, DepthEstimationPipelineConfig
from pipeline.camera.camera import Camera, EvaluationCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipelineContext
from pipeline.depth_estimation_pipeline_hooks import DepthEstimationPipelineHook
from pipeline.depth_estimation_pipeline_metrics import DepthEstimationPipelineMetric


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


def reduce_metrics(metrics_results: Dict[str, List[float]], reduction: Literal["mean", "sum"]) -> Dict[str, float]:
    _reduction_ops = {
        "mean": lambda x: sum(x) / len(x),
        "sum": sum
    }
    return {
        key: _reduction_ops[reduction](value) for key, value in metrics_results.items()
    }


def run_depth_estimation_pipeline(camera: Camera,
                                  pipeline: DepthEstimationPipeline,
                                  hooks: Iterable[DepthEstimationPipelineHook] = None) -> None:
    if hooks is None:
        hooks = []

    pipeline_config = pipeline.get_configuration()
    validate_pipeline_config_wrt_camera(pipeline_config, camera)

    n_parallel_jobs = min(len(hooks), cpu_count() - 1, 1)
    with Parallel(n_jobs=n_parallel_jobs) as parallel_thread_pool:
        parallel_thread_pool(delayed(hook.on_pipeline_start)() for hook in hooks)

        for frame_index, (left_view, right_view) in enumerate(camera.stream_image_pairs()):
            pipeline_result = pipeline.process(left_view, right_view)

            pipeline_context = DepthEstimationPipelineContext(
                disparity_map=pipeline_result.disparity_map,
                left_image=pipeline_result.left_image,
                right_image=pipeline_result.right_image,
                config=pipeline_config,
                frame_index=frame_index
            )

            parallel_thread_pool(
                delayed(DepthEstimationPipelineHook.invoke_in_context)(hook, pipeline_context) for hook in hooks
            )

        parallel_thread_pool(delayed(hook.on_pipeline_end)() for hook in hooks)


def run_depth_estimation_pipeline_evaluation(camera: EvaluationCamera,
                                             pipeline: DepthEstimationPipeline,
                                             metrics: Iterable[DepthEstimationPipelineMetric] = None,
                                             reduction: Literal["mean", "sum"] = "mean",
                                             verbose: bool = True) -> Dict[str, float]:
    if metrics is None:
        metrics = []

    metrics_results = {metric.name(): [] for metric in metrics}
    max_disp = pipeline.get_configuration().max_disparity

    validate_pipeline_config_wrt_camera(pipeline.get_configuration(), camera)

    for frame_index, (left_view, right_view, gt_disparity) in enumerate(camera.stream_image_pairs_with_gt_disparity()):
        gt_disparity = gt_disparity.cuda()
        pipeline_result = pipeline.process(left_view, right_view)
        gt_mask = (gt_disparity <= max_disp) & (gt_disparity > 0)

        for metric in metrics:
            metric_loss = metric.process(pipeline_result.disparity_map, gt_disparity, gt_mask)
            metrics_results[metric.name()].append(metric_loss)

        if verbose:
            print(f"Processed frame {frame_index}.")

    return reduce_metrics(metrics_results, reduction)

