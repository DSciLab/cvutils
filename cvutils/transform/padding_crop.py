from typing import List, Tuple, Union
import numpy as np
from .base import Transformer
from .crop import RandomCenterCrop, RandomCrop
from .padding import ZeroPadding
from .utils import size_cmp


class _ZeroPaddingCrop(Transformer):
    def __init__(
        self,
        padding_size: Union[List[int], Tuple[int, int], int],
        crop_size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__()
        assert size_cmp(padding_size, crop_size) <= 0, \
            f'padding_size < crop_size, padding_size={padding_size},'\
            f' crop_size={crop_size}'

        if isinstance(padding_size, int):
            self.padding_size = [padding_size, padding_size]
        else:
            self.padding_size = padding_size

        if isinstance(crop_size, int):
            self.crop_size = [crop_size, crop_size]
        else:
            self.crop_size = crop_size

        self.padding_fn = ZeroPadding(self.padding_size)

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        inp = self.padding_fn(inp)
        inp = self.crop_fn(inp)
        return inp


class ZeroPaddingRandomCrop(_ZeroPaddingCrop):
    def __init__(
        self,
        padding_size: Union[List[int], Tuple[int, int], int],
        crop_size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__(padding_size, crop_size)
        self.crop_fn = RandomCrop(crop_size)


class ZeroPaddingRandomCenterCrop(_ZeroPaddingCrop):
    def __init__(
        self,
        padding_size: Union[List[int], Tuple[int, int], int],
        crop_size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__(padding_size, crop_size)
        self.crop_fn = RandomCenterCrop(crop_size)
