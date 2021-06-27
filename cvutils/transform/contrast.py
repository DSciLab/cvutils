from typing import Optional, Tuple, Union, List
import random
import numpy as np
from .base import Transformer


def contrast(
    inp: np.ndarray,
    contrast_range: Optional[Union[Tuple[float, float], List[float]]]=(0.75, 1.25),
) -> np.ndarray:

    if np.random.random() < 0.5 and contrast_range[0] < 1:
        factor = np.random.uniform(contrast_range[0], 1)
    else:
        factor = np.random.uniform(max(contrast_range[0], 1),
                                   contrast_range[1])

    for c in range(inp.shape[0]):
        mn = inp[c].mean()
        minm = inp[c].min()
        maxm = inp[c].max()

        inp[c] = (inp[c] - mn) * factor + mn

        inp[c][inp[c] < minm] = minm
        inp[c][inp[c] > maxm] = maxm
    return inp


class RandomContrast(Transformer):
    def __init__(
        self,
        contrast_range: Optional[Union[Tuple[float, float], List[float]]]=(0.75, 1.25),
    ) -> None:
        super().__init__()
        self.contrast_range = contrast_range

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        return contrast(inp, self.contrast_range)
