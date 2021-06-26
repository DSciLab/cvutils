from typing import Union
import torch
from torch import Tensor
import numpy as np
from .base import Transformer


class ToTensor(Transformer):
    def __call__(self, inp: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(inp, Tensor):
            return inp
        else:
            inp = torch.from_numpy(inp)
        return inp
