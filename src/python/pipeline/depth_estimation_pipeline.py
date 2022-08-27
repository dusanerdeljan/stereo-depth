from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Any

import torch
import cuda_depth

from helpers.paths import python_project_relative_path
from helpers.torch_helpers import cuda_perf_clock
from pipeline.depth import StereoMatching, DnnStereoMatchingBackend, CudaStereoMatchingBackend
from pipeline.synthesis import RightViewSynthesis


@dataclass
class DepthEstimationPipelineConfig:
    image_shape: Tuple[int, int] = (384, 1280)
    min_disparity: int = 1
    max_disparity: int = 64
    invalid_disparity: float = -1.0
    dnn_stereo_matching_path = python_project_relative_path("data/traced/traced_MSNet2D.pt")
    stereo_matching_backend: Literal["dnn", "cuda"] = "cuda"

    def update(self, **kwargs: Any) -> DepthEstimationPipelineConfig:
        for (key, value) in kwargs.items():
            if not hasattr(self, key):
                raise RuntimeError(f"Unexpected keyword argument: '{key}'.")
            setattr(self, key, value)
        return self


@dataclass
class DepthEstimationPipelineContext:
    disparity_map: torch.Tensor
    left_image: torch.Tensor
    right_image: torch.Tensor
    config: DepthEstimationPipelineConfig
    frame_index: int


class DepthEstimationPipeline:
    
    def __init__(self, config: DepthEstimationPipelineConfig = DepthEstimationPipelineConfig()) -> None:
        self._config = config
        self._right_view_synthesis = RightViewSynthesis()
        self._stereo_matching = self._get_stereo_matching()
        print(f"Using '{self._config.stereo_matching_backend}' as stereo matching backend.")

    def process(self, left_image: torch.Tensor, right_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        left_image = left_image.cuda()
        with cuda_perf_clock("Right view generation"):
            if right_image is None:
                right_image = self._right_view_synthesis.process(left_image)
        with cuda_perf_clock("Stereo matching"):
            disparity_map = self._stereo_matching.process(left_image, right_image)
        return disparity_map, (left_image, right_image)

    def get_configuration(self) -> DepthEstimationPipelineConfig:
        return self._config

    def _get_stereo_matching(self) -> StereoMatching:
        if self._config.stereo_matching_backend == "dnn":
            stereo_matching = DnnStereoMatchingBackend(
                traced_model_path=self._config.dnn_stereo_matching_path
            )
            return stereo_matching
        elif self._config.stereo_matching_backend == "cuda":
            config = cuda_depth.StereoMatchingConfiguration(
                height=self._config.image_shape[0],
                width=self._config.image_shape[1],
                min_disparity=self._config.min_disparity,
                max_disparity=self._config.max_disparity
            )
            stereo_matching = CudaStereoMatchingBackend(configuration=config)
            return stereo_matching
        else:
            raise RuntimeError(f"Unsupported stereo matching backend: {self._config.stereo_matching_backend}")
