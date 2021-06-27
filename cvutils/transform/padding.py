from typing import List, Tuple, Union
import numpy as np
from .base import Transformer


class _Padding(Transformer):
    def __init__(
        self,
        size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size


class ZeroPadding(_Padding):
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        C, H, W = inp.shape
        if H > self.size[0] and W > self.size[1]:
            return inp

        output_h = max(H, self.size[0])
        output_w = max(W, self.size[1])

        output = np.zeros([C, output_h, output_w], dtype=inp.dtype)
        padding_start_h = (output_h - H) // 2
        padding_start_w = (output_w - W) // 2

        output[:, padding_start_h: padding_start_h + H,
               padding_start_w: padding_start_w + W] = inp

        return output
