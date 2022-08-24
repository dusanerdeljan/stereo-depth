from pipeline.camera import MiddleBuryStereoCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipeline
from pipeline.depth_estimation_pipeline_runner import run_depth_estimation_pipeline, extract_config_from_camera
from pipeline.disparity_map_hooks import DisparityMapSaver, PointCloudSaver

if __name__ == "__main__":
    camera = MiddleBuryStereoCamera(
        middlebury_dir="data"
    )
    depth_pipeline = DepthEstimationPipeline(config=extract_config_from_camera(camera))
    run_depth_estimation_pipeline(
        camera=camera,
        pipeline=depth_pipeline,
        hooks=[
            DisparityMapSaver(save_dir="data/temp"),
            #PointCloudSaver.for_camera(camera=camera, save_dir="data/temp", invalid_disparity=-1.0),
        ]
    )

