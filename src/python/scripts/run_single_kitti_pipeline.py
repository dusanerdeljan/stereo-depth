from pipeline.camera import KittiSingleViewCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipeline
from pipeline.depth_estimation_pipeline_runner import run_depth_estimation_pipeline, extract_config_from_camera
from pipeline.disparity_map_hooks import DisparityMapCompletionLogger, DisparityMapSaver

if __name__ == "__main__":
    camera = KittiSingleViewCamera(
        drive_dir="data/train/2011_09_26/2011_09_26_drive_0011_sync",
        return_right_view=False,
        only_one=True
    )
    depth_pipeline = DepthEstimationPipeline(config=extract_config_from_camera(camera))
    run_depth_estimation_pipeline(
        camera=camera,
        pipeline=depth_pipeline,
        hooks=[DisparityMapCompletionLogger(), DisparityMapSaver(save_dir="data/temp")]
    )
