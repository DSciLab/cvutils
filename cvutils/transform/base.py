import numpy as np


class Transformer(object):
    def __init__(self) -> None:
        pass

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError
