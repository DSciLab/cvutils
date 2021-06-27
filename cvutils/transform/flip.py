import random
import numpy as np
from .base import Transformer


class FlipX(Transformer):
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """
        :param inp: shape (C, Y, X)
        """
        return np.flip(inp, axis=2)


class FlipY(Transformer):
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """
        :param inp: shape (C, Y, X)
        """
        return np.flip(inp, axis=1)


class RandomFlip(Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.flip_fn = [FlipX(), FlipY()]

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        flip_fn = random.choice(self.flip_fn)
        return flip_fn(inp)
