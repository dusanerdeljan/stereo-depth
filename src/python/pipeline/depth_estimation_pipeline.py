from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np
# noinspection PyUnresolvedReferences
import torch

from helpers.paths import python_project_relative_path
from pipeline.depth import StereoMatching, DnnStereoMatchingBackend, CudaStereoMatchingBackend
from pipeline.synthesis import RightViewSynthesis


@dataclass
class DepthEstimationPipelineConfig:
    image_shape: Tuple[int, int] = (384, 1280)
    min_disparity: int = 1
    max_disparity: int = 16
    invalid_disparity: float = -1.0
    dnn_stereo_matching_path = python_project_relative_path("data/traced/traced_MSNet3D.pt")
    stereo_matching_backend: Literal["dnn", "cuda"] = "cuda"


class DepthEstimationPipeline:
    
    def __init__(self, config: DepthEstimationPipelineConfig = DepthEstimationPipelineConfig()) -> None:
        self._config = config
        self._right_view_synthesis = RightViewSynthesis()
        self._stereo_matching = self._get_stereo_matching()
        print(f"Using '{self._config.stereo_matching_backend}' as stereo matching backend.")

    def process(self, left_image: np.ndarray, right_image: Optional[np.ndarray] = None) -> np.ndarray:
        if right_image is None:
            right_image = self._right_view_synthesis.process(left_image)
        disparity_map = self._stereo_matching.process(left_image, right_image)
        return disparity_map

    def get_configuration(self) -> DepthEstimationPipelineConfig:
        return self._config

    def _get_stereo_matching(self) -> StereoMatching:
        if self._config.stereo_matching_backend == "dnn":
            stereo_matching = DnnStereoMatchingBackend(
                traced_model_path=self._config.dnn_stereo_matching_path
            )
            return stereo_matching
        elif self._config.stereo_matching_backend == "cuda":
            stereo_matching = CudaStereoMatchingBackend()
            return stereo_matching
        else:
            raise RuntimeError(f"Unsupported stereo matching backend: {self._config.stereo_matching_backend}")
