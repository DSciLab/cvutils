from typing import List, Tuple, Union
import numpy as np
from .base import Transformer
from .crop import RandomCenterCrop, RandomCrop
from .resize import Resize
from .utils import size_cmp


class _ResizeCrop(Transformer):
    def __init__(
        self,
        resize_size: Union[List[int], Tuple[int, int], int],
        crop_size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__()
        assert size_cmp(resize_size, crop_size) <= 0, \
            f'resize_size < crop_size, resize_size={resize_size},'\
            f' crop_size={crop_size}'

        if isinstance(resize_size, int):
            self.resize_size = [resize_size, resize_size]
        else:
            self.resize_size = resize_size

        if isinstance(crop_size, int):
            self.crop_size = [crop_size, crop_size]
        else:
            self.crop_size = crop_size

        self.resize = Resize()

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        resize_to = self.resize_size
        inp_shape = inp.shape

        resize_to[0] = max(resize_to[0], inp_shape[1])
        resize_to[1] = max(resize_to[1], inp_shape[2])

        inp = self.resize.resize_by(inp, resize_to)
        inp = self.crop_fn(inp)
        return inp


class ResizeRandomCrop(_ResizeCrop):
    def __init__(
        self,
        resize_size: Union[List[int], Tuple[int, int], int],
        crop_size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__(resize_size, crop_size)
        self.crop_fn = RandomCrop(crop_size)


class ResizeRandomCenterCrop(_ResizeCrop):
    def __init__(
        self,
        resize_size: Union[List[int], Tuple[int, int], int],
        crop_size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__(resize_size, crop_size)
        self.crop_fn = RandomCenterCrop(crop_size)
