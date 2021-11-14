from typing import List, Tuple, Union
import numpy as np
from .base import Transformer
from .crop import RandomCenterCrop, RandomCrop
from .resize import Resize


class _CropResize(Transformer):
    def __init__(
        self,
        final_size: Union[List[int], Tuple[int, int], int],
        crop_range: Union[List[float], Tuple[float, float]]
    ) -> None:
        super().__init__()
        assert len(crop_range) == 2 and\
            crop_range[1] <= 1.0 and\
                crop_range[0] > 0 and\
                    crop_range[0] < crop_range[1]

        self.crop_range = crop_range
        self.resize = Resize(final_size)

    def get_rand_crop_ratio(self) -> float:
        return np.random.uniform(*self.crop_range)

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        crop_ratio = self.get_rand_crop_ratio()
        inp = self.crop.crop_by_ratio(inp, crop_ratio)
        inp = self.resize(inp)
        return inp


class RandomCropResize(_CropResize):
    def __init__(
        self,
        final_size: Union[List[int], Tuple[int, int], int],
        crop_range: Union[List[float], Tuple[float, float]]
    ) -> None:
        super().__init__(final_size, crop_range)
        self.crop = RandomCrop(None)


class RandomCenterCropResize(_CropResize):
    def __init__(
        self,
        final_size: Union[List[int], Tuple[int, int], int],
        crop_range: Union[List[float], Tuple[float, float]]
    ) -> None:
        super().__init__(final_size, crop_range)
        self.crop = RandomCenterCrop(None)
