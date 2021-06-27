
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage.interpolation import rotate
from .base import Transformer


class RandomRotate(Transformer):
    def __init__(
        self,
        rotate_range: Optional[Union[List[float],
                                     Tuple[float, float]]]=[0.0, np.pi / 4]
    ) -> None:
        super().__init__()
        self.rotate_range = rotate_range

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

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        theta = np.random.uniform() * (self.rotate_range[1] - self.rotate_range[0])\
                + self.rotate_range[0]

        height = inp.shape[1]
        width = inp.shape[2]
        affine_matrix = self.transform_matric(theta, width, height)

        inp_ = []
        for i in range(inp.shape[0]):
            inp_.append(affine_transform(inp[i], affine_matrix))
        inp = np.stack(inp_, axis=0)

        return inp
