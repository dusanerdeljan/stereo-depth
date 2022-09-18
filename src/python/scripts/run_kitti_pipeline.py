from typing import Literal

from helpers.paths import python_project_relative_path, timestamp_folder_name
from pipeline.camera import KittiSingleViewCamera
from pipeline.depth_estimation_pipeline import DepthEstimationPipeline
from pipeline.depth_estimation_pipeline_hooks import ContextFrameSaver, ContextVideoSaver
from pipeline.depth_estimation_pipeline_runner import run_depth_estimation_pipeline, extract_config_from_camera


_backends_to_fps = {
    "cuda": 30,
    "gwcnet": 6,
    "msnet3d": 4
}


def run_single_stereo_backend(stereo_backend: Literal["cuda", "gwcnet", "msnet3d"]):
    camera = KittiSingleViewCamera(
        drive_dir="data/train/2011_09_26/2011_09_26_drive_0084_sync",
        return_right_view=False,
        only_one=False
    )
    config = extract_config_from_camera(camera).update(stereo_matching_backend=stereo_backend)
    depth_pipeline = DepthEstimationPipeline(config=config)
    video_name = f"{stereo_backend}_{timestamp_folder_name()}.mp4"
    run_depth_estimation_pipeline(
        camera=camera,
        pipeline=depth_pipeline,
        hooks=[
            ContextFrameSaver(save_dir=python_project_relative_path("data/temp")),
            ContextVideoSaver(
                save_path=python_project_relative_path(f"data/temp/videos/{video_name}"),
                fps=_backends_to_fps[stereo_backend]
            )
        ]
    )


if __name__ == "__main__":
    for stereo_matching_backend in _backends_to_fps.keys():
        run_single_stereo_backend(stereo_matching_backend)
