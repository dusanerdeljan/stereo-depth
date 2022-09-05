from abc import ABC, abstractmethod

import torch


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
        return torch.mean(err_mask.float()).cpu()

    def name(self) -> str:
        return "D1"
