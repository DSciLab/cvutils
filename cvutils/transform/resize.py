from typing import Optional, List, Tuple, Union
import random
import numpy as np
from scipy.ndimage import affine_transform
from .base import Transformer


class Resize(Transformer):
    def __init__(self) -> None:
        pass

    def transform_matric(self, scale: List[float]) -> np.ndarray:
        assert len(scale) == 2, f'len(sclae) = {len(scale)} != 2'

        resize_axis_matrix = np.array(
            [[1 / scale[0],     0.,            0.],
             [0.,          1 / scale[1],       0.],
             [0.,               0.,            1.]])

        return resize_axis_matrix

    def __call__(
        self,
        inp: np.ndarray,
        scale: Optional[float]=None,
        size: Union[Tuple[int, int], List[int], int]=None
    ) -> np.ndarray:
        assert scale is not None or size is not None, \
            'Scale is None and size is None.'
        assert scale is None or size is None, \
            'Ambiguous, scale is not None and size is not None.'

        height = inp.shape[1]
        width = inp.shape[2]

        if scale is not None and not isinstance(scale, (tuple, list)):
            scale = (scale, scale)

        if size is not None and not isinstance(size, (tuple, list)):
            size = (size, size)

        if scale is None:
            scale = (size[0] / height,
                     size[1] / width)

        if size is None:
            size = (int(height * scale[0]),
                    int(width * scale[1]))

        affine_matrix = self.transform_matric(scale)
        if inp.ndim == 2:
            # gray image
            inp = affine_transform(
                inp, affine_matrix, output_shape=size)
        else:
            # RGB image
            inp_ = []
            for i in range(inp.shape[0]):
                inp_.append(affine_transform(
                    inp[i], affine_matrix, output_shape=size))
            inp = np.stack(inp_, axis=0)

        return inp
