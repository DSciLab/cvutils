from typing import List, Tuple, Union, Optional
import numpy as np


def size_cmp(
    size1: Union[List[int], Tuple[int, int], Tuple[int, int, int], int],
    size2: Union[List[int], Tuple[int, int], Tuple[int, int, int], int]
) -> int:
    """
    if size1 > size2, return -1
    elif size1 == size2, return 0
    elif size1 < size2, return 1
    """
    assert type(size1) is type(size2)

    if isinstance(size1, int):
        if size1 > size2: return -1
        elif size1 < size2: return 1
        else: return 0
    else:
        assert len(size1) == len(size2)
        for i in range(len(size1)):
            if size1[i] > size2[i]:
                return -1
            elif size1[i] < size2[i]:
                return 1
            else:
                continue

        return 0


def get_range_val(
    value: Union[list, tuple, float],
    rand_type: Optional[str]='uniform'
) -> float:
    if isinstance(value, (list, tuple)):
        if rand_type == 'uniform':
            value = np.random.uniform(value[0], value[1])
        elif rand_type == 'normal':
            value = np.random.normal(value[0], value[1])
        else:
            raise ValueError(
                f'Unrecognized rand_type ({rand_type})')
        return value
    else:
        return value