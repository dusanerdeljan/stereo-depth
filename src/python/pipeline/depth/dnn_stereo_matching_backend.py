import torch
import torchvision.transforms as T

from helpers.paths import python_project_relative_path
from pipeline.depth import StereoMatching

AVAILABLE_DNN_BACKENDS = ["msnet2d", "msnet3d", "gwcnet"]

_traced_model_paths = {
    "msnet2d": python_project_relative_path("data/traced/traced_MSNet2D.pt"),
    "msnet3d": python_project_relative_path("data/traced/traced_MSNet3D.pt"),
    "gwcnet": python_project_relative_path("data/traced/traced_gwcnet-g.pt")
}


def _preprocess(image: torch.Tensor) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocessing = T.Compose([
        T.Lambda(lambda t: t.cuda().float() / 255.0),
        T.Normalize(mean=mean, std=std),
        T.Lambda(lambda t: torch.unsqueeze(t, 0))
    ])
    return preprocessing(image)


class DnnStereoMatchingBackend(StereoMatching):

    def __init__(self, traced_model: str):
        if traced_model not in _traced_model_paths:
            raise RuntimeError(f"Unsupported DNN stereo backend traced model: '{traced_model}'.")
        self._dnn_inference = torch.jit.load(_traced_model_paths[traced_model])

    @torch.no_grad()
    def process(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        left_gpu = _preprocess(left_image)
        right_gpu = _preprocess(right_image)
        return self._dnn_inference(left_gpu, right_gpu).squeeze(0)
