from typing import List, Tuple, Union
import numpy as np
from numpy.core.fromnumeric import resize
from .base import Transformer
from .crop import RandomCenterCrop, RandomCrop
from .resize import Resize
from .padding import ZeroPadding


class _ResizeCroPad(Transformer):
    def __init__(
        self,
        final_size: Union[List[int], Tuple[int, int], int],
        rand_range:Union[List[int], Tuple[int, int]] =(0.5, 1.5)
    ) -> None:
        super().__init__()
        assert rand_range[1] >= rand_range[0]

        self.range = rand_range
        if isinstance(final_size, int):
            self.final_size = [final_size, final_size]
        else:
            self.final_size = final_size

        self.resize = Resize()
        self.pad = ZeroPadding(final_size)

    def get_rand(self) -> float:
        return np.random.uniform(*self.range)

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        inp_shape = inp.shape
        resize_scale = self.get_rand()
        resize_to = [0, 0]

        resize_to[0] = int(inp_shape[1] * resize_scale)
        resize_to[1] = int(inp_shape[2] * resize_scale)

        inp = self.resize.resize_by(inp, resize_to)
        if resize_scale >= 1.0:
            inp = self.crop_fn(inp)
        else:
            inp = self.pad(inp)

        return inp


class ResizeRandomCroPad(_ResizeCroPad):
    def __init__(
        self,
        final_size: Union[List[int], Tuple[int, int], int],
        rand_range:Union[List[int], Tuple[int, int]] =(0.5, 1.5)
    ) -> None:
        super().__init__(final_size, rand_range)
        self.crop_fn = RandomCrop(final_size)


class ResizeRandomCenterCroPad(_ResizeCroPad):
    def __init__(
        self,
        final_size: Union[List[int], Tuple[int, int], int],
        rand_range:Union[List[int], Tuple[int, int]] =(0.5, 1.5)
    ) -> None:
        super().__init__(final_size, rand_range)
        self.crop_fn = RandomCenterCrop(final_size)
