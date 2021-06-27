import numpy as np
from .base import Transformer


class FixChannels(Transformer):
    @staticmethod
    def maybe_transpose_torch(
        inp: np.ndarray
    ) -> np.ndarray:
        """
        for image with channels
        if channels is 1, maybe the inp is a gray image
        if channels is 3, maybe the inp is a RGB image
        """
        inp_shape = inp.shape
        if inp_shape[0] in [1, 3]:
            return inp
        elif inp_shape[2] in [1, 3]:
            return inp.transpose(2, 0, 1)
        else:
            raise ValueError(
                f'Unrecognized image shape ({inp_shape}).')

    def __call__(
        self,
        inp: np.ndarray
    ) -> np.ndarray:
        if inp.ndim == 3:
            inp = self.maybe_transpose_torch(inp)
            return inp
        elif inp.ndim == 2:
            return np.expand_dims(inp, axis=0)
        else:
            raise ValueError(
                f'Unrecognized input dimension, inp.ndim={inp.ndim}')
