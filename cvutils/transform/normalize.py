from typing import List, Optional, Union
import numpy as np
from .base import Transformer


class LinearNormalize(Transformer):
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        inp_max = inp.max()
        inp_min = inp.min()
        inp = (inp - inp_min) / (inp_max - inp_min)
        return inp


class Normalize(Transformer):
    def __init__(
        self,
        mean: Optional[Union[List[int], int]]=None,
        std: Optional[Union[List[int], int]]=None
    ) -> None:
        super().__init__()
        if (mean is not None or std is not None) and not \
            (mean is not None and std is not None):
            raise ValueError(f'Argument error. mean={mean}, std={std}')

        if mean is not None and std is not None:
            self.mean = np.array(mean)
            self.std = np.array(std)
        else:
            self.mean = None
            self.std = None

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        C = inp.shape[0]
        if self.mean is not None and self.std is not None:
            mean = self.mean
            std = self.std
        else:
            mean = inp.reshape(C, -1).mean(1)
            std = inp.reshape(C, -1).std(1)

        mean = mean.reshape(C, 1, 1)
        std = std.reshape(C, 1, 1)

        inp = (inp - mean) / std
        return inp
