import itertools
from typing import Literal, Dict, Any

from pipeline.camera import KittiSingleViewCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipeline
from pipeline.depth_estimation_pipeline_metrics import D1Metric, ThresholdMetric, MAEMetric
from pipeline.depth_estimation_pipeline_runner import extract_config_from_camera, \
    run_depth_estimation_pipeline_evaluation


def evaluate_depth_estimation_pipeline(
        drive_dir: str,
        use_right_view_synthesis: bool,
        depth_estimation_backend: Literal["msnet2d", "msnet3d", "gwcnet", "cuda"],
) -> Dict[str, float]:
    camera = KittiSingleViewCamera(
        drive_dir=drive_dir,
        return_right_view=not use_right_view_synthesis,
        only_one=True
    )
    config = extract_config_from_camera(camera).update(stereo_matching_backend=depth_estimation_backend)
    depth_pipeline = DepthEstimationPipeline(config=config)
    metrics = run_depth_estimation_pipeline_evaluation(
        camera=camera,
        pipeline=depth_pipeline,
        metrics=[
            D1Metric(),
            ThresholdMetric(1.0),
            ThresholdMetric(2.0),
            ThresholdMetric(3.0),
            ThresholdMetric(5.0),
            MAEMetric()
        ],
        reduction="mean",
        verbose=False
    )
    return metrics


def print_metrics(metrics: Dict[str, float], **kwargs: Any) -> None:
    print("================================================================")
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    print("")
    print("{:<15} {:<8}".format("Metric name", "Metric value"))
    for metric_name, metric_value in metrics.items():
        print("{:<15} {:<8}".format(metric_name, metric_value))
    print("================================================================")


if __name__ == "__main__":
    drives = ["data/train/2011_09_26/2011_09_26_drive_0019_sync", "data/train/2011_09_26/2011_09_26_drive_0084_sync"]
    use_right_view_synthesis_values = [True, False]
    depth_estimation_backend_values = ["cuda", "gwcnet", "msnet2d", "msnet3d"]
    # Can't run this in parallel due to high GPU memory requirements :(
    args_iterator = itertools.product(drives, use_right_view_synthesis_values, depth_estimation_backend_values)
    for (drive, use_rvs, depth_backend) in args_iterator:
        metric = evaluate_depth_estimation_pipeline(
            drive_dir=drive,
            use_right_view_synthesis=use_right_view_synthesis_values,
            depth_estimation_backend=depth_backend
        )
        print_metrics(
            metrics=metric,
            drive=drive,
            use_right_view_synthesis=use_rvs,
            depth_estimation_backend=depth_backend
        )
