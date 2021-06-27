import numpy as np
from .base import Transformer


class TransposeTorch(Transformer):
    def __call__(
        self,
        inp: np.ndarray
    ) -> np.ndarray:
        if inp.ndim == 2:
            # for gray image
            return inp
        else:
            # for RGB image
            inp_shape = inp.shape
            if inp_shape[0] == 3:
                return inp
            else:
                # the shape of inp is (H, W, C)
                # to (C, H, W)
                return inp.transpose(2, 0, 1)


class UntransposeTorch(Transformer):
    def __call__(
        self,
        inp: np.ndarray
    ) -> np.ndarray:
        if inp.ndim == 2:
            # for gray image
            return inp
        else:
            # for RGB image
            inp_shape = inp.shape
            if inp_shape[2] == 3:
                return inp
            else:
                # the shape of inp is (C, H, W)
                # to (H, W, C)
                return inp.transpose(1, 2, 0)
