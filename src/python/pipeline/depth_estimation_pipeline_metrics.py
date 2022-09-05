from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class DepthEstimationPipelineMetric(ABC):

    @abstractmethod
    def process(self, disparity_estimate: torch.Tensor, disparity_gt: torch.Tensor, mask: torch.Tensor) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class D1Metric(DepthEstimationPipelineMetric):

    def process(self, disparity_estimate: torch.Tensor, disparity_gt: torch.Tensor, mask: torch.Tensor) -> float:
        disparity_estimate = disparity_estimate[mask]
        disparity_gt = disparity_gt[mask]
        E = torch.abs(disparity_estimate - disparity_gt)
        err_mask = (E > 3) & (E / disparity_gt.abs() > 0.05)
        return torch.mean(err_mask.float()).item()

    def name(self) -> str:
        return "D1"


class ThresholdMetric(DepthEstimationPipelineMetric):

    def __init__(self, threshold: float):
        super(ThresholdMetric, self).__init__()
        self._threshold = threshold

    def process(self, disparity_estimate: torch.Tensor, disparity_gt: torch.Tensor, mask: torch.Tensor) -> float:
        disparity_estimate = disparity_estimate[mask]
        disparity_gt = disparity_gt[mask]
        E = torch.abs(disparity_estimate - disparity_gt)
        err_mask = E > self._threshold
        return torch.mean(err_mask.float()).item()

    def name(self) -> str:
        return f"Threshold_{int(self._threshold)}"


class MAEMetric(DepthEstimationPipelineMetric):

    def process(self, disparity_estimate: torch.Tensor, disparity_gt: torch.Tensor, mask: torch.Tensor) -> float:
        disparity_estimate = disparity_estimate[mask]
        disparity_gt = disparity_gt[mask]
        return F.l1_loss(disparity_estimate, disparity_gt).item()

    def name(self) -> str:
        return "MAE"
