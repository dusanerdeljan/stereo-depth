from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helpers.torch_helpers import get_best_available_device, \
    initialize_conv2d, get_vgg_conv_blocks, initialize_linear


class DeconvBlock(nn.Module):

    def __init__(self, filter_size: int, scale: int, deconv_kernel_size: int):
        super(DeconvBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_size, filter_size, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(filter_size, filter_size, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.deconv = nn.ConvTranspose2d(filter_size, 65, kernel_size=(deconv_kernel_size, deconv_kernel_size),
                                         stride=(scale, scale), padding=(scale // 2, scale // 2))

        initialize_conv2d(self.conv1)
        initialize_conv2d(self.conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class ConvDeconvBlock(nn.Module):

    def __init__(self, conv_block: nn.Module, filter_size: int, deconv_kernel_size: int, scale: int):
        super(ConvDeconvBlock, self).__init__()
        self.conv_block = conv_block
        self.deconv_block = DeconvBlock(
            filter_size=filter_size,
            scale=scale,
            deconv_kernel_size=deconv_kernel_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv_block(x)
        deconv_out = self.deconv_block(conv_out)
        return conv_out, deconv_out


class ConvDeconvStack(nn.Module):

    def __init__(self, deconv_filter_sizes: List[int], backbone_vgg: nn.Module):
        super(ConvDeconvStack, self).__init__()
        self._scale = 1

        backbone_vgg_conv_blocks = get_vgg_conv_blocks(backbone_vgg)
        if len(deconv_filter_sizes) != len(backbone_vgg_conv_blocks):
            raise RuntimeError("Invalid deconv filter size provided.")

        self._num_blocks = len(deconv_filter_sizes)
        for idx, conv_block in enumerate(backbone_vgg_conv_blocks):
            self._scale = self._scale if idx == 0 else self._scale * 2
            kernel_size = 1 if idx == 0 else self._scale * 2
            conv_deconv_block = ConvDeconvBlock(
                conv_block=conv_block,
                filter_size=deconv_filter_sizes[idx],
                deconv_kernel_size=kernel_size,
                scale=self._scale
            )
            self.register_module(f"conv_deconv_{idx}", conv_deconv_block)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        disparity_predictions = []
        features = x
        for idx in range(self._num_blocks):
            conv_deconv_block = self.get_submodule(f"conv_deconv_{idx}")
            features, prediction = conv_deconv_block(features)
            disparity_predictions.append(prediction)
        return features, disparity_predictions

    @property
    def scale(self) -> int:
        return self._scale


class DisparityUpconvSoftmax(nn.Module):

    def __init__(self):
        super(DisparityUpconvSoftmax, self).__init__()
        self.deconv = nn.ConvTranspose2d(65, 65, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv = nn.Conv2d(65, 65, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        initialize_conv2d(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = F.relu(x)
        x = self.conv(x)
        x = F.softmax(x, dim=1)
        return x


class FeedForwardBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        initialize_linear(self.fc1)
        initialize_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DisparityEstimationNetwork(nn.Module):

    def __init__(self,
                 deconv_filter_sizes: List[int],
                 backbone_vgg: nn.Module = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT),
                 ff_input_dim: int = 15360,
                 ff_hidden_dim: int = 4096,
                 ff_output_dim: int = 1950):
        super(DisparityEstimationNetwork, self).__init__()
        self.conv_deconv_stack = ConvDeconvStack(
            deconv_filter_sizes=deconv_filter_sizes,
            backbone_vgg=backbone_vgg
        )

        self.feed_forward = FeedForwardBlock(
            input_dim=ff_input_dim,
            hidden_dim=ff_hidden_dim,
            output_dim=ff_output_dim
        )

        scale = self.conv_deconv_stack.scale
        self.disparity_deconv = nn.ConvTranspose2d(65, 65, kernel_size=(scale * 2, scale * 2),
                                                   stride=(scale, scale), padding=(scale // 2, scale // 2))

        self.softmax_upconv = DisparityUpconvSoftmax()

    def forward(self, x: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        x, disparity_predictions = self.conv_deconv_stack(x)

        x = self.feed_forward(x.view(original_shape[0], -1))
        x = self.disparity_deconv(x.view(original_shape[0], 65, 3, 10))

        disparity_predictions.append(x)
        disparity_prediction = torch.sum(torch.stack(disparity_predictions), dim=0)
        disparity_prediction = self.softmax_upconv(disparity_prediction)
        return F.interpolate(disparity_prediction, scale_factor=4, mode="bilinear")

    @property
    def scale(self) -> int:
        return self._scale


class ViewSynthesisNetwork(nn.Module):

    def __init__(self, device: torch.device):
        super(ViewSynthesisNetwork, self).__init__()
        self._device = device

    def forward(self, disparity_estimate: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
        shifted_views = [self._shift_view(-depth_map_index, view) for depth_map_index in range(65)]
        stacked_shifted_view = torch.stack(shifted_views, dim=1)
        mult_soft_shift_out = torch.mul(disparity_estimate.unsqueeze(2), stacked_shifted_view)
        generated_right_view = torch.sum(mult_soft_shift_out, dim=1)
        return generated_right_view

    def _shift_view(self, depth_map_index: int, view: torch.Tensor) -> torch.Tensor:
        shifted_view = torch.zeros_like(view, device=self._device)
        if depth_map_index < 0:
            shifted_view[:, :, :, :depth_map_index] = view[:, :, :, -depth_map_index:]
        elif depth_map_index == 0:
            shifted_view = view
        else:
            shifted_view[:, :, :, depth_map_index:] = view[:, :, :, :-depth_map_index]
        return shifted_view


class Deep3D(nn.Module):

    def __init__(self,
                 device: torch.device = torch.device("cuda"),
                 backbone_vgg: nn.Module = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT),
                 log_perf_time: bool = False):
        super(Deep3D, self).__init__()
        self._log_perf = log_perf_time
        self._device = device

        self.disparity_estimation_network = DisparityEstimationNetwork(
            deconv_filter_sizes=[64, 128, 256, 512, 512],
            backbone_vgg=backbone_vgg,
            ff_input_dim=15360,
            ff_hidden_dim=4096,
            ff_output_dim=1950
        )

        self.view_synthesis_network = ViewSynthesisNetwork(
            device=device
        )

    def forward(self, left_full_resolution: torch.Tensor, left_downscaled: torch.Tensor) -> torch.Tensor:
        disparity_estimate = self.disparity_estimation_network(left_downscaled, left_full_resolution.shape)
        synthesized_view = self.view_synthesis_network(disparity_estimate, left_full_resolution)
        return synthesized_view

    @property
    def device(self):
        return self._device


def test_model_view_generation():
    device = get_best_available_device()
    model = Deep3D(device=device, log_perf_time=True).to(device)

    # CUDA warmup
    model(torch.randn(1, 3, 384, 1280, dtype=torch.float32, device=device),
          torch.randn(1, 3, 96, 320, dtype=torch.float32, device=device))
    torch.cuda.synchronize()

    for _ in range(100):
        generated_view = model(torch.randn(1, 3, 384, 1280, dtype=torch.float32, device=device),
                               torch.randn(1, 3, 96, 320, dtype=torch.float32, device=device))
        print(generated_view.shape)


if __name__ == '__main__':
    test_model_view_generation()
