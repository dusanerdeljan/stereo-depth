import itertools
import json
import os
from typing import Literal, Dict, List

import torch.cuda

from helpers.paths import timestamp_folder_name
from pipeline.camera import KittiSingleViewCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipeline
from pipeline.depth_estimation_pipeline_metrics import D1Metric, ThresholdMetric, MAEMetric
from pipeline.depth_estimation_pipeline_runner import extract_config_from_camera, \
    run_depth_estimation_pipeline_evaluation


def evaluate_depth_estimation_pipeline_single_parameters(
        drive_dir: str,
        use_right_view_synthesis: bool,
        depth_estimation_backend: Literal["msnet2d", "msnet3d", "gwcnet", "cuda"],
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    camera = KittiSingleViewCamera(
        drive_dir=drive_dir,
        return_right_view=not use_right_view_synthesis,
        only_one=False
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


def evaluate_depth_estimation_pipeline(
        drives: List[str],
        save_dir: str
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    use_right_view_synthesis_values = [True, False]
    depth_estimation_backend_values = ["cuda", "gwcnet", "msnet3d"]
    # Can't run this in parallel due to high GPU memory requirements :(
    args_iterator = itertools.product(drives, use_right_view_synthesis_values, depth_estimation_backend_values)
    json_metrics = []
    for (drive, use_rvs, depth_backend) in args_iterator:
        metric = evaluate_depth_estimation_pipeline_single_parameters(
            drive_dir=drive,
            use_right_view_synthesis=use_rvs,
            depth_estimation_backend=depth_backend
        )
        json_metrics.append(dict(
            drive=drive,
            use_right_view_synthesis=use_rvs,
            depth_estimation_backend=depth_backend,
            metrics=metric,
        ))
    timestamp_file_name = f"evaluation_{timestamp_folder_name()}.json"
    with open(os.path.join(save_dir, timestamp_file_name), "w") as json_file:
        json.dump(json_metrics, json_file, indent=4)


if __name__ == "__main__":
    evaluate_depth_estimation_pipeline(
        drives=["data/train/2011_09_26/2011_09_26_drive_0019_sync", "data/train/2011_09_26/2011_09_26_drive_0084_sync"],
        save_dir="../data/temp/eval_sessions"
    )
