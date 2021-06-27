from typing import Optional, Tuple, Union
import random
import numpy as np
from .base import Transformer


def augment_gamma(
    inp: np.ndarray,
    *,
    gamma_range: Optional[Tuple[float, float]]=(0.5, 2.0),
    invert_image: Optional[bool]=False,
    epsilon: Optional[bool]=1e-7,
    retain_stats: Optional[bool]=False
) -> np.ndarray:

    if invert_image:
        inp = - inp

    # For per channels
    for c in range(inp.shape[0]):
        if retain_stats:
            mn = inp[c].mean()
            sd = inp[c].std()

        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])

        minm = inp[c].min()
        rnge = inp[c].max() - minm
        inp[c] = np.power(((inp[c] - minm) / float(rnge + epsilon)), gamma)\
                    * float(rnge + epsilon) + minm

        if retain_stats:
            inp[c] = inp[c] - inp[c].mean()
            inp[c] = inp[c] / (inp[c].std() + 1e-8) * sd
            inp[c] = inp[c] + mn

    if invert_image:
        inp = - inp
    return inp


class RandomGamma(Transformer):
    def __init__(
        self,
        gamma_range: Optional[Tuple[float, float]]=(0.5, 2.0),
        invert_image: Optional[bool]=False,
        retain_stats : Optional[bool]=False
    ) -> None:

        super().__init__()
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.retain_stats = retain_stats

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:

        return augment_gamma(
            inp,
            gamma_range=self.gamma_range,
            invert_image=self.invert_image,
            retain_stats=self.retain_stats)
