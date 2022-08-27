import torch
import torchvision.transforms as T

from pipeline.depth import StereoMatching


def preprocess(image: torch.Tensor) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocessing = T.Compose([
        T.Lambda(lambda t: t.cuda().float() / 255.0),
        T.Normalize(mean=mean, std=std),
        T.Lambda(lambda t: torch.unsqueeze(t, 0))
    ])
    return preprocessing(image)


class DnnStereoMatchingBackend(StereoMatching):

    def __init__(self, traced_model_path: str):
        self._dnn_inference = torch.jit.load(traced_model_path)

    @torch.no_grad()
    def process(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        left_gpu = preprocess(left_image)
        right_gpu = preprocess(right_image)
        return self._dnn_inference(left_gpu, right_gpu).squeeze(0)
