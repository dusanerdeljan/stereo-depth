from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Optional

import torch


class Camera(ABC):

    @abstractmethod
    def focal_length(self) -> float:
        pass

    @abstractmethod
    def baseline(self) -> float:
        pass

    @abstractmethod
    def get_image_shape(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_disparity_boundaries(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def stream_image_pairs(self) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        pass
