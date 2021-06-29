from typing import List, Optional, Union
import numpy as np
from torch import Tensor
from numpy.core.fromnumeric import shape
import torch
from .base import Transformer


class DeLinearNormalize(Transformer):
    def __init__(
        self,
        max_value: Optional[float]=255.0
    ) -> None:
        super().__init__()
        self.max_value = max_value

    def tensor(self, inp: Tensor) -> Tensor:
        return torch.clamp(inp * self.max_value,
                           min=0.0, max=255.0).type(torch.int)

    def numpy(self, inp: np.ndarray) -> np.ndarray:
        return np.clip(inp * self.max_value,
                       a_min=0.0, a_max=255.0).astype(np.int)

    def __call__(
        self,
        inp: Union[np.ndarray, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        if isinstance(inp, Tensor):
            return self.tensor(inp)
        else:
            return self.numpy(inp)


class DeNormalize(Transformer):
    def __init__(
        self,
        mean: Union[List[int], int],
        std: Union[List[int], int]
    ) -> None:
        super().__init__()
        self.mean = np.array(mean)
        self.std = np.array(std)

    def tensor(self, inp: Tensor) -> Tensor:
        if inp.ndim == 2:
            # for gray image
            mean = self.mean.reshape(1, 1)
            std = self.std.reshape(1, 1)
        elif inp.ndim == 3:
            # for RGB image or single channel image
            d1, _, d3 = inp.shape
            if d1 in [1, 3]:
                # image shape is (C, H, W)
                mean = self.mean.reshape(d1, 1, 1)
                std = self.std.reshape(d1, 1, 1)
            elif d3 in [1, 3]:
                # image shape is (H, W, C)
                mean = self.mean.reshape(1, 1, d3)
                std = self.std.reshape(1, 1, d3)
            else:
                raise ValueError(
                    f'Unrecognized image shape {inp.shape}')
        elif inp.ndim == 4:
            # for RGB image or single channel image with batch
            _, d1, _, d3 = inp.shape
            if d1 in [1, 3]:
                # image shape is (C, H, W)
                mean = self.mean.reshape(1, d1, 1, 1)
                std = self.std.reshape(1, d1, 1, 1)
            elif d3 in [1, 3]:
                # image shape is (H, W, C)
                mean = self.mean.reshape(1, 1, 1, d3)
                std = self.std.reshape(1, 1, 1, d3)
            else:
                raise ValueError(
                    f'Unrecognized image shape {inp.shape}')
        else:
            raise ValueError(
                f'Unrecognized image dimension inp.ndim={inp.ndim}')

        mean = torch.from_numpy(mean).to(inp.device)
        std = torch.from_numpy(std).to(inp.device)
        inp = torch.clamp(inp * std + mean,
                          min=0.0, max=255.0).type(torch.int)
        return inp

    def numpy(self, inp: np.ndarray) -> np.ndarray:
        if inp.ndim == 2:
            # for gray image
            mean = self.mean.reshape(1, 1)
            std = self.std.reshape(1, 1)
        elif inp.ndim == 3:
            # for RGB image or single channel image
            d1, _, d3 = inp.shape
            if d1 in [1, 3]:
                # image shape is (C, H, W)
                mean = self.mean.reshape(d1, 1, 1)
                std = self.std.reshape(d1, 1, 1)
            elif d3 in [1, 3]:
                # image shape is (H, W, C)
                mean = self.mean.reshape(1, 1, d3)
                std = self.std.reshape(1, 1, d3)
            else:
                raise ValueError(
                    f'Unrecognized image shape {inp.shape}')
        elif inp.ndim == 4:
            # for RGB image or single channel image with batch
            _, d1, _, d3 = inp.shape
            if d1 in [1, 3]:
                # image shape is (C, H, W)
                mean = self.mean.reshape(1, d1, 1, 1)
                std = self.std.reshape(1, d1, 1, 1)
            elif d3 in [1, 3]:
                # image shape is (H, W, C)
                mean = self.mean.reshape(1, 1, 1, d3)
                std = self.std.reshape(1, 1, 1, d3)
            else:
                raise ValueError(
                    f'Unrecognized image shape {inp.shape}')
        else:
            raise ValueError(
                f'Unrecognized image dimension inp.ndim={inp.ndim}')

        inp = np.clip(inp * std + mean,
                      a_min=0.0, a_max=255.0).astype(np.int)
        return inp

    def __call__(
        self,
        inp: Union[np.ndarray, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        if isinstance(inp, Tensor):
            return self.tensor(inp)
        else:
            return self.numpy(inp)
