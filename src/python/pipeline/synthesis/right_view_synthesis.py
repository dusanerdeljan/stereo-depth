import torch
import cuda_synthesis
import torchvision.io
import torchvision.transforms as T

from helpers.paths import DEEP3D_MODEL_TRACE_PATH


class RightViewSynthesis:
    
    def __init__(self):
        super(RightViewSynthesis, self).__init__()
        self._device = torch.device("cuda")
        self._full_resolution = (384, 1280)
        self._downscaled_resolution = (384 // 4, 1280 // 4)
        self._dnn_inference = cuda_synthesis.RightViewSynthesis(
            height=self._full_resolution[0], width=self._full_resolution[1], model_path=DEEP3D_MODEL_TRACE_PATH
        )

        self._resize_to_full_resolution = T.Resize(size=self._full_resolution)
        self._resize_to_downscaled_resolution = T.Resize(size=self._downscaled_resolution)

        # Perform CUDA warmup for faster inference
        self._cuda_warmup()

    @torch.no_grad()
    def process(self, left_view: torch.Tensor) -> torch.Tensor:
        left_full_resolution = self._resize_to_full_resolution(left_view).cuda() / 255.0
        left_downscaled = self._resize_to_downscaled_resolution(left_view).cuda() / 255.0

        generated_right_view = self._dnn_inference.generate_right_view(left_full_resolution, left_downscaled)
        return generated_right_view

    @torch.no_grad()
    def _cuda_warmup(self) -> None:
        self._dnn_inference.generate_right_view(
            torch.randn(3, *self._full_resolution, dtype=torch.float32, device=self._device),
            torch.randn(3, *self._downscaled_resolution, dtype=torch.float32, device=self._device)
        )
        torch.cuda.synchronize()


def test_right_view_synthesis() -> None:
    right_view_synthesis = RightViewSynthesis()
    left_view = torchvision.io.read_image("../../data/train/2011_09_26/2011_09_26_drive_0011_sync/image_02/data"
                                          "/0000000000.png")
    right_view = right_view_synthesis.process(left_view.cuda())
    torchvision.io.write_png(left_view.cpu(), "../../data/temp/generated_left_view.png")
    torchvision.io.write_png(right_view.byte().cpu(), "../../data/temp/generated_right_view.png")


if __name__ == "__main__":
    test_right_view_synthesis()
