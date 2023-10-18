__all__ = ['SharedDict']

import pickle
from collections.abc import Iterator, Mapping
from typing import TypeVar

import numpy as np
import torch

_K = TypeVar('_K')
_V = TypeVar('_V')


class SharedDict(Mapping[_K, _V]):
    """
    Mapping that holds its values in shared memory via `torch.Tensor`.
    """
    __slots__ = ('_keys', '_buf', '_addr')

    def __init__(self, obj: Mapping[_K, _V]) -> None:
        self._keys = {p: i for i, p in enumerate(obj)}

        vs = [_serialize(v) for v in obj.values()]
        self._buf = torch.from_numpy(np.concatenate([*vs]))
        self._addr = torch.as_tensor([0] + [len(v) for v in vs]).cumsum(0)

    def __getitem__(self, key: _K) -> _V:
        idx = self._keys[key]
        lo, hi = self._addr[idx:idx + 2].tolist()
        buf = memoryview(self._buf[lo:hi].numpy())
        return pickle.loads(buf)

    def __iter__(self) -> Iterator[_K]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def _serialize(v) -> np.ndarray:
    buf = pickle.dumps(v, protocol=-1)
    return np.frombuffer(buf, dtype='u1')
