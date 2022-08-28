from helpers.paths import python_project_relative_path
from pipeline.camera import KittiSingleViewCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipeline
from pipeline.depth_estimation_pipeline_hooks import ContextFrameSaver
from pipeline.depth_estimation_pipeline_runner import run_depth_estimation_pipeline, extract_config_from_camera

if __name__ == "__main__":
    camera = KittiSingleViewCamera(
        drive_dir="data/train/2011_09_26/2011_09_26_drive_0011_sync",
        return_right_view=False,
        only_one=False
    )
    depth_pipeline = DepthEstimationPipeline(config=extract_config_from_camera(camera))
    run_depth_estimation_pipeline(
        camera=camera,
        pipeline=depth_pipeline,
        hooks=[ContextFrameSaver(save_dir=python_project_relative_path("data/temp"))]
    )
