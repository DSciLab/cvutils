from typing import List, Tuple, Union
import numpy as np
from .base import Transformer
from .crop import RandomCenterCrop, RandomCrop
from .resize import Resize
from .padding import ZeroPadding


class _CropResize(Transformer):
    def __init__(
        self,
        final_size: Union[List[int], Tuple[int, int], int],
        crop_range: Union[List[float], Tuple[float, float]]
    ) -> None:
        super().__init__()
        assert len(crop_range) == 2 and\
                crop_range[0] > 0 and\
                    crop_range[0] < crop_range[1]

        self.crop_range = crop_range
        self.resize = Resize(final_size)

    def get_rand_crop_ratio(self) -> float:
        return np.random.uniform(*self.crop_range)

    def pad(
        self,
        inp: np.ndarray,
        ratio: float
    ) -> np.ndarray:
        assert ratio > 1.0
        C, H, W = inp.shape
        output_h, output_w = int(ratio * H), int(ratio * W)

        output = np.zeros([C, output_h, output_w], dtype=inp.dtype)
        padding_start_h = (output_h - H) // 2
        padding_start_w = (output_w - W) // 2

        output[:, padding_start_h: padding_start_h + H,
               padding_start_w: padding_start_w + W] = inp
        return output

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        crop_ratio = self.get_rand_crop_ratio()
        if crop_ratio > 1.0:
            inp = self.pad(inp, crop_ratio)
        else:
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
