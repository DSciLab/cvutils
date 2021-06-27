import re
from typing import List, Tuple, Union
import numpy as np
from .base import Transformer


def random_crop(
    inp: np.ndarray,
    size: Union[List[int], Tuple[int, int], int]
) -> np.ndarray:
    """
    :param inp: shape (C, H, W)
    """
    if isinstance(size, int):
        size = [size, size]

    inp_shape = inp.shape
    H = inp_shape[1]
    W = inp_shape[2]

    assert size[0] <= H, f'size on input is {inp_shape}, crop size is {size}.'
    assert size[1] <= W, f'size on input is {inp_shape}, crop size is {size}.'

    if size[0] == H and size[1] == W:
        return inp

    rand_h_start = int((H - size[0]) * np.random.uniform())
    rand_w_start = int((W - size[1]) * np.random.uniform())

    return inp[:, rand_h_start: rand_h_start + size[0],
               rand_w_start: rand_w_start + size[1]]


def random_center_crop(
    inp: np.ndarray,
    size: Union[List[int], Tuple[int, int], int]
) -> np.ndarray:
    """
    :param inp: shape (C, H, W)
    """
    if isinstance(size, int):
        size = [size, size]

    inp_shape = inp.shape
    H = inp_shape[1]
    W = inp_shape[2]

    assert size[0] <= H,\
        f'size on input is {inp_shape}, crop size is {size}.'
    assert size[1] <= W,\
        f'size on input is {inp_shape}, crop size is {size}.'

    if size[0] == H and size[1] == W:
        return inp

    half_shift_h = (H - size[0]) // 2
    half_shift_w = (W - size[1]) // 2

    rand_h_start = int(half_shift_h +
                       half_shift_h * np.clip(np.random.normal(loc=0, scale=0.5),
                                              a_min=-1.0, a_max=1.0))
    rand_w_start = int(half_shift_w +
                       half_shift_w * np.clip(np.random.normal(loc=0, scale=0.5),
                                              a_min=-1.0, a_max=1.0))

    return inp[:, rand_h_start: rand_h_start + size[0],
               rand_w_start: rand_w_start + size[1]]


class RandomCrop(Transformer):
    def __init__(
        self,
        size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size
    
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return random_crop(inp, self.size)


class RandomCenterCrop(Transformer):
    def __init__(
        self,
        size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size
    
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return random_center_crop(inp, self.size)
