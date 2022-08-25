import torch
import cuda_depth

from pipeline.depth import StereoMatching


class CudaStereoMatchingBackend(StereoMatching):

    def __init__(self,
                 configuration: cuda_depth.StereoMatchingConfiguration = cuda_depth.StereoMatchingConfiguration()):
        self._stereo_algo = cuda_depth.StereoMatching(configuration)

    def process(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        left_gpu = left_image.cuda().float().contiguous()
        right_gpu = right_image.cuda().float().contiguous()
        output_disparity = self._stereo_algo.compute_disparity_map(left_gpu, right_gpu)
        return output_disparity
