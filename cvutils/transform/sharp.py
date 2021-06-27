from typing import Optional, Tuple, Union
import random
import numpy as np
from scipy.signal import convolve
from .base import Transformer


class RandomSharpening(Transformer):
    filter_2d = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])

    def __init__(
        self,
        strength: Optional[Union[float, Tuple[float, float]]]=[0.1, 1.0],
    ) -> None:
        self.strength = strength

    def augument_sharp(
        self,
        inp: np.ndarray
    ) -> np.ndarray:

        for c in range(inp.shape[0]):
            mn, mx = inp[c, :, :].min(), inp[c, :, :].max()

            if isinstance(self.strength, float):
                strength_here = self.strength
            else:
                strength_here = np.random.uniform(*self.strength)
            filter_here = self.filter_2d * strength_here

            inp[c, :, :] = convolve(inp[c, :, :], filter_here, mode='same') * 0.4\
                           + inp[c, :, :] * 0.6
            inp[c, :, :] = np.clip(inp[c, :, :], mn, mx)
        return inp

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        return self.augument_sharp(inp)
