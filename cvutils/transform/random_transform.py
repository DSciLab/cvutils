from typing import List, Optional, Tuple, Union
import numpy as np
import random

from scipy.ndimage import interpolation
from .base import Transformer

from .flip import RandomFlip
from .rotate import RandomRotate
from .brightness import RandomBrightness
from .contrast import RandomContrast
from .gamma import RandomGamma
from .gaussian_blur import RandomGaussianBlur
from .median_filter import RandomMedianFilter
from .noise import RandomNoise
from .padding_crop import ZeroPaddingRandomCenterCrop
from .resize_crop import ResizeRandomCenterCrop
from .sharp import RandomSharpening
from .normalize import Normalize


__all__ = ['RandomTransform']


class OrOpration(Transformer):
    def __init__(self, ops: List[Transformer]) -> None:
        super().__init__()
        self.ops = ops

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        op = random.choice(self.ops)
        return op(inp)


class RandomTransform(Transformer):
    """
    USE ME AFTER NORNALIZE.
    """
    def __init__(
        self,
        k: int,
        input_size: Union[List[int], Tuple[int, int], int]
    ) -> None:
        super().__init__()
        self.k = k
        self.color_ops = [
            RandomBrightness(),
            RandomContrast(),
            RandomGamma(),
            RandomGaussianBlur(),
            RandomMedianFilter(),
            RandomNoise(),
            RandomSharpening()
        ]
        self.spatial_ops = [
            RandomFlip(),
            ResizeRandomCenterCrop(input_size, input_size)
        ]
        assert self.k <= len(self.color_ops),\
            f'k should be less than {len(self.color_ops)}, k={self.k}.'

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        # check inp normalize
        if not Normalize.is_normalized(inp):
            print(f'[WARNING] input data is not normalized.')

        selected_color_ops = random.sample(self.color_ops, k=self.k)

        for op in selected_color_ops:
            try:
                inp = op(inp)
            except Exception as e:
                print(f'[DEBUG] Transformer '
                      f'{op.__class__.__name__} error.'
                      f'inp shape: {inp.shape}, '
                      f'inp.max={inp.max()}, '
                      f'inp.min={inp.min()}')
                raise e

        for op in self.spatial_ops:
            try:
                inp = op(inp)
            except Exception as e:
                print(f'[DEBUG] Transformer '
                      f'{op.__class__.__name__} error.'
                      f'inp shape: {inp.shape}, '
                      f'inp.max={inp.max()}, '
                      f'inp.min={inp.min()}')
                raise e

        return inp
