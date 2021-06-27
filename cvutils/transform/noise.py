from typing import List, Optional, Tuple, Union
import numpy as np
import random
from .base import Transformer


class NoiseBase(Transformer):
    """
        Use me after normalization.
    """
    def __init__(
        self,
        noise_variance: Optional[Union[Tuple[float, float], List[float]]]=(0, 0.1)
    ) -> None:
        super().__init__()
        self.noise_variance = noise_variance

    def augment_noise(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        return self.augment_noise(inp)


class RandomRicianNoise(NoiseBase):
    def __init__(
        self,
        noise_variance: Optional[Union[Tuple[float, float], List[float]]]=(0, 0.3)
    ) -> None:
        super().__init__(noise_variance)

    def augment_noise(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        if self.noise_variance[0] == self.noise_variance[1]:
            variance = self.noise_variance[0]
        else:
            variance = random.uniform(self.noise_variance[0], self.noise_variance[1])

        inp_sign = np.sign(inp)
        inp = np.sqrt(
            (inp + np.random.normal(0.0, variance, size=inp.shape)) ** 2 +
             np.random.normal(0.0, variance, size=inp.shape) ** 2)

        inp = inp * inp_sign

        return inp


class RandomGaussianNoise(NoiseBase):
    def __init__(
        self,
        noise_variance: Optional[Union[Tuple[float, float], List[float]]]=(0, 0.15)
    ) -> None:
        super().__init__(noise_variance)

    def augment_noise(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        if self.noise_variance[0] == self.noise_variance[1]:
            variance = self.noise_variance[0]
        else:
            variance = random.uniform(self.noise_variance[0], self.noise_variance[1])

        inp = inp + np.random.normal(0.0, variance, size=inp.shape)
        return inp


class RandomNoise(Transformer):
    """
        Use me after normalization.
    """
    def __init__(
        self,
        noise_variance: Optional[Union[Tuple[float, float], List[float]]]=(0, 0.1)
    ) -> None:
        super().__init__()
        self.noise_fn = [RandomRicianNoise(noise_variance),
                         RandomGaussianNoise(noise_variance)]

    def __call__(
        self,
        inp: np.ndarray,
    ) -> np.ndarray:
        noise_fn = random.choice(self.noise_fn)
        return noise_fn(inp)
