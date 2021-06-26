
from typing import Union
import numpy as np
from scipy.ndimage import affine_transform
from .base import Transformer


class Rotate(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(
        self,
        theta: float,
        width: int,
        height: int
    ) -> np.ndarray:
        move_axis_matrix = np.array(
            [[1.,     0.,   width / 2.],
             [0.,     1.,  height / 2.],
             [0.,     0.,           1.]])

        move_axis_matrix_back = np.array(
            [[1.,     0.,   -width / 2.],
             [0.,     1.,  -height / 2.],
             [0.,     0.,            1.]])

        rotate_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0.],
             [np.sin(theta), np.cos(theta),  0.],
             [0.,            0.,             1.]])

        return move_axis_matrix @ rotate_matrix @ move_axis_matrix_back

    def __call__(self, inp: np.ndarray, theta: float) -> np.ndarray:
        assert inp.ndim == 3, f'input dim error inp.ndim={inp.ndim}'

        height = inp.shape[1]
        width = inp.shape[2]
        affine_matrix = self.transform_matric(theta, width, height)

        inp_ = []
        for i in range(inp.shape[0]):
            inp_.append(affine_transform(inp[i], affine_matrix))
        inp = np.stack(inp_, axis=0)

        return inp


class RandomRotate(Transformer):
    def __init__(
        self,
        r_min: Union[int, float],
        r_max: Union[int, float]
    ) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'

        self.r_max = r_max
        self.r_min = r_min
        self.rotater = Rotate()

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.rotater(inp, theta)
