import numpy as np
import torch
from torchvision import transforms

from pipeline.depth import StereoMatching


def preprocess(image: np.ndarray) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocessing = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return preprocessing(image)


class DnnStereoMatchingBackend(StereoMatching):

    def __init__(self, traced_model_path: str):
        self._dnn_inference = torch.jit.load(traced_model_path)

    @torch.no_grad()
    def process(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        left_gpu = preprocess(left_image).cuda().unsqueeze(0)
        right_gpu = preprocess(right_image).cuda().unsqueeze(0)
        return self._dnn_inference(left_gpu, right_gpu).squeeze(0).cpu().numpy()
