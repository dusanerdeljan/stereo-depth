import os

from joblib import Parallel, delayed

from pipeline.camera import MiddleBuryStereoCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipeline
from pipeline.depth_estimation_pipeline_hooks import ContextFrameSaver, DisparityMapCompletionLogger
from pipeline.depth_estimation_pipeline_runner import run_depth_estimation_pipeline, extract_config_from_camera


def process_single_middlebury_frame(base_dir: str, frame_dir: str) -> None:
    camera = MiddleBuryStereoCamera(
        middlebury_dir=os.path.join(base_dir, frame_dir)
    )
    depth_pipeline = DepthEstimationPipeline(config=extract_config_from_camera(camera))
    run_depth_estimation_pipeline(
        camera=camera,
        pipeline=depth_pipeline,
        hooks=[DisparityMapCompletionLogger(), ContextFrameSaver(save_dir=f"data/temp/{frame_dir}")]
    )


if __name__ == "__main__":
    data_dir = "../data/middlebury/data"
    with Parallel() as parallel_thread_pool:
        parallel_thread_pool(
            delayed(process_single_middlebury_frame)(data_dir, frame_dir) for frame_dir in os.listdir(data_dir)
        )
