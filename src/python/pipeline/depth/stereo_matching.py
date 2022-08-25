from abc import ABC, abstractmethod

import torch


class StereoMatching(ABC):

    @abstractmethod
    def process(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        pass

