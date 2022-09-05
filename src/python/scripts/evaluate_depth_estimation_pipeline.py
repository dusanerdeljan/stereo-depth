from pipeline.camera import KittiSingleViewCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipeline
from pipeline.depth_estimation_pipeline_metrics import D1Metric
from pipeline.depth_estimation_pipeline_runner import extract_config_from_camera, \
    run_depth_estimation_pipeline_evaluation

if __name__ == "__main__":
    camera = KittiSingleViewCamera(
        drive_dir="data/train/2011_09_26/2011_09_26_drive_0019_sync",
        return_right_view=False,
        only_one=False
    )
    depth_pipeline = DepthEstimationPipeline(config=extract_config_from_camera(camera))
    metrics = run_depth_estimation_pipeline_evaluation(
        camera=camera,
        pipeline=depth_pipeline,
        metrics=[D1Metric()],
        reduction="mean",
        verbose=True
    )
    print(metrics)
