import numpy as np


class Transformer(object):
    def __init__(self) -> None:
        self.strength = 1.0

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_strength(self) -> None:
        pass

    def call_apply_strength(self, strength: float) -> None:
        self.strength = strength
        self.apply_strength()
