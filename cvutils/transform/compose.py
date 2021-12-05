from typing import List
import numpy as np
from .base import Transformer


class Compose(Transformer):
    def __init__(
        self,
        transformer_list: List[Transformer]
    ) -> None:
        super().__init__()
        # check transformer
        for transformer in transformer_list:
            assert isinstance(transformer, Transformer), \
                f'transformer error, type of '\
                f'{transformer} is {type(transformer)}.'
        self.transformer_list = transformer_list
    
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        for transformer in self.transformer_list:
            try:
                inp = transformer(inp)
            except Exception as e:
                print(
                    f'[DEBUG] Transformer '
                    f'{transformer.__class__.__name__} error. '
                    f'inp shape: {inp.shape}, '
                    f'inp.max={inp.max()}, '
                    f'inp.min={inp.min()}'
                )
                raise e
        return inp

    def apply_strength(self) -> None:
        for transformer in self.transformer_list:
            try:
                transformer.call_apply_strength(self.strength)
            except Exception as e:
                print(
                    f'[DEBUG] Transformer '
                    f'({transformer.__class__.__name__}) apply strength. '
                    f'strength: {self.strength}'
                )
                raise e
