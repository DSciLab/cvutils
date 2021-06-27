from typing import Optional, Tuple, Union
import numpy as np
import random
from .base import Transformer


class _BrightnessBase(Transformer):

    def __call__(self,
        inp: np.ndarray,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        inp = self.augment_brightness(inp)
        return inp


class RandomBrightnessAdditive(_BrightnessBase):
    def __init__(
        self,
        mu: float=0.0,
        sigma: float=0.2
    ) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def augment_brightness(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        for c in range(inp.shape[0]):
            rnd_nb = np.random.normal(self.mu, self.sigma)
            inp[c] += rnd_nb
        return inp


class RandomBrightnessMultiplicative(_BrightnessBase):
    def __init__(
        self,
        multiplier_range: Optional[Tuple[float, float]]=(0.5, 2)
    ) -> None:
        super().__init__()
        self.multiplier_range = multiplier_range

    def augment_brightness(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:

        for c in range(inp.shape[0]):
            multiplier = np.random.uniform(self.multiplier_range[0],
                                           self.multiplier_range[1])
            inp[c] *= multiplier
        return inp


class RandomBrightness(_BrightnessBase):
    def __init__(self) -> None:
        super().__init__()
        self.brightness_fn = [
            RandomBrightnessAdditive(), RandomBrightnessMultiplicative()]

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        brightness_fn = random.choice(self.brightness_fn)
        if brightness_fn is None:
            return inp
        else:
            return brightness_fn(inp)
