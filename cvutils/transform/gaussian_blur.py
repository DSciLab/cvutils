from typing import List, Optional, Tuple, Union
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from .base import Transformer
from .utils import get_range_val


def gaussian_blur(
    inp: np.ndarray,
    sigma: Optional[Union[List[int], float]]=[0, 0.8]
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    sigma = get_range_val(sigma)
    return gaussian_filter(inp, sigma=sigma)


class RandomGaussianBlur(Transformer):
    def __init__(
        self,
        sigma: Optional[Union[List[int], float]]=[0.0, 1.0]
    ) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:

        if self.sigma is not None:
            return gaussian_blur(inp, self.sigma)
        else:
            return gaussian_blur(inp)
