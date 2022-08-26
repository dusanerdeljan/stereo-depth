import torch
import torchvision.transforms as T

from pipeline.depth import StereoMatching


def preprocess(image: torch.Tensor) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocessing = T.Compose([T.Normalize(mean=mean, std=std)])
    return preprocessing(image)


class DnnStereoMatchingBackend(StereoMatching):

    def __init__(self, traced_model_path: str):
        self._dnn_inference = torch.jit.load(traced_model_path)

    @torch.no_grad()
    def process(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        left_gpu = preprocess(left_image.float()).cuda().unsqueeze(0)
        right_gpu = preprocess(right_image.float()).cuda().unsqueeze(0)
        return self._dnn_inference(left_gpu, right_gpu).squeeze(0)
