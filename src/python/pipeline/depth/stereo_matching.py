from abc import ABC, abstractmethod

import numpy as np


class StereoMatching(ABC):

    @abstractmethod
    def process(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        pass

