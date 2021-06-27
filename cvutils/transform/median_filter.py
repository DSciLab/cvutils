from typing import Optional, Tuple, Union
import random
import numpy as np
from scipy.ndimage import median_filter
from .base import Transformer


class RandomMedianFilter(Transformer):
    def __init__(
        self,
        filter_size: Union[int, Tuple[int, int]]=(1, 3),
    ) -> None:
        self.filter_size = filter_size

    def augument_median_filter(
        self,
        inp: np.ndarray
    ) -> np.ndarray:

        for c in range(inp.shape[0]):
            if isinstance(self.filter_size, int):
                filter_size = self.filter_size
            else:
                filter_size = np.random.randint(*self.filter_size)
            inp[c, :, :] = median_filter(inp[c, :, :], filter_size)
        return inp

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        return self.augument_median_filter(inp)
