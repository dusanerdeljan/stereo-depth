import numpy as np
import torch
import cuda_synthesis

from helpers.paths import DEEP3D_MODEL_TRACE_PATH
from helpers.torch_helpers import get_best_available_device
from helpers.imageio_helpers import to_resized_torch_image, load_cv_rgb_image, resize_image, save_cv_rgb_image


class RightViewSynthesis:
    
    def __init__(self):
        super(RightViewSynthesis, self).__init__()
        self._device = get_best_available_device()
        self._full_resolution = (384, 1280)
        self._downscaled_resolution = (384 // 4, 1280 // 4)
        self._dnn_inference = cuda_synthesis.RightViewSynthesis(
            height=self._full_resolution[0], width=self._full_resolution[1], model_path=DEEP3D_MODEL_TRACE_PATH
        )

        # Perform CUDA warmup for faster inference
        self._cuda_warmup()

    @torch.no_grad()
    def process(self, left_view: np.ndarray) -> np.ndarray:
        left_full_resolution = to_resized_torch_image(left_view, self._full_resolution).to(self._device).float()
        left_downscaled = to_resized_torch_image(left_view, self._downscaled_resolution).to(self._device).float()

        generated_right_view = self._dnn_inference.generate_right_view(left_full_resolution, left_downscaled)
        return generated_right_view.to("cpu", torch.uint8).numpy()

    @torch.no_grad()
    def _cuda_warmup(self):
        self._dnn_inference.generate_right_view(
            torch.randn(3, *self._full_resolution, dtype=torch.float32, device=self._device),
            torch.randn(3, *self._downscaled_resolution, dtype=torch.float32, device=self._device)
        )
        torch.cuda.synchronize()


def test_right_view_synthesis():
    full_resolution = (384, 1280)
    right_view_synthesis = RightViewSynthesis()
    left_view = load_cv_rgb_image("../../data/train/2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000000.png")
    right_view = right_view_synthesis.process(left_view)
    left_rescaled = np.clip(255 * resize_image(left_view, full_resolution) + 0.5, 0, 255).astype(np.uint8)
    save_cv_rgb_image("../../data/temp/generated_left_view.png", left_rescaled)
    save_cv_rgb_image("../../data/temp/generated_right_view.png", right_view)
    print("Saved generated image...")


if __name__ == "__main__":
    test_right_view_synthesis()
