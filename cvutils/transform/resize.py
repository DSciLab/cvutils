from typing import Optional, List, Tuple, Union
import random
import numpy as np
from numpy.core.fromnumeric import amin
from scipy.ndimage import affine_transform
from .base import Transformer


class Resize(Transformer):
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int], List[int]]]=None
    ) -> None:
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size

    def transform_matric(self, scale: List[float]) -> np.ndarray:
        assert len(scale) == 2, f'len(sclae) = {len(scale)} != 2'

        resize_axis_matrix = np.array(
            [[1 / scale[0],     0.,            0.],
             [0.,          1 / scale[1],       0.],
             [0.,               0.,            1.]])

        return resize_axis_matrix

    def resize_by(
        self,
        inp: np.ndarray,
        size: Optional[Union[int, Tuple[int, int], List[int]]]=None
    ) -> np.ndarray:
        if isinstance(size, int):
            size = [size, size]
        else:
            size = size

        height = inp.shape[1]
        width = inp.shape[2]

        scale = (size[0] / height,
                 size[1] / width)

        affine_matrix = self.transform_matric(scale)
        inp_ = []
        for i in range(inp.shape[0]):

            c_inp_min = inp[i].min()
            c_inp_max = inp[i].max()
            c_inp = affine_transform(
                inp[i], affine_matrix, output_shape=size)
            c_inp = np.clip(c_inp, a_min=c_inp_min, a_max=c_inp_max)
            inp_.append(c_inp)

        inp = np.stack(inp_, axis=0)
        return inp

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        if self.size in None:
            raise ValueError(f'target size is None.')

        height = inp.shape[1]
        width = inp.shape[2]

        scale = (self.size[0] / height,
                 self.size[1] / width)

        affine_matrix = self.transform_matric(scale)
        inp_ = []
        for i in range(inp.shape[0]):

            c_inp_min = inp[i].min()
            c_inp_max = inp[i].max()
            c_inp = affine_transform(
                inp[i], affine_matrix, output_shape=self.size)
            c_inp = np.clip(c_inp, a_min=c_inp_min, a_max=c_inp_max)
            inp_.append(c_inp)

        inp = np.stack(inp_, axis=0)

        return inp
