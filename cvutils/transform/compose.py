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
                print(f'[DEBUG] Transformer '
                      f'{transformer.__class__.__name__} error.')
                raise e
        return inp
