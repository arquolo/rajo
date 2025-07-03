__all__ = ['SharedDict', 'SharedList']

import pickle
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import overload

import numpy as np
import torch
from torch import Tensor


class SharedList[T](Sequence[T]):
    """
    Mapping that holds its values in shared memory via `torch.Tensor`.
    """

    __slots__ = ('_addr', '_buf')

    def __init__(self, items: Iterable[T]) -> None:
        ts = [_serialize(x) for x in items]
        self._buf = torch.cat(ts) if ts else torch.empty(0, dtype=torch.uint8)
        self._addr = torch.as_tensor([0] + [len(t) for t in ts]).cumsum(0)

    @overload
    def __getitem__(self, index: int, /) -> T: ...

    @overload
    def __getitem__(self, index: slice, /) -> list[T]: ...

    def __getitem__(self, index: int | slice, /) -> T | list[T]:
        len_ = len(self)
        if isinstance(index, slice):
            return [self[i] for i in range(len_)[index]]

        if not -len_ <= index < len_:
            raise IndexError(f'{type(self).__name__} index out of range')
        index %= len_
        lo, hi = self._addr[index : index + 2].tolist()
        return _deserialize(self._buf[lo:hi])

    def __iter__(self) -> Iterator[T]:
        if not self:
            return iter(())
        return map(_deserialize, self._buf.tensor_split(self._addr[1:-1]))

    def __len__(self) -> int:
        return self._addr.shape[0] - 1

    def __repr__(self) -> str:
        return f'{type(self).__name__}(len={len(self)})'


class SharedDict[K, V](Mapping[K, V]):
    """
    Mapping that holds its values in shared memory via `torch.Tensor`.
    """

    __slots__ = ('_keys', '_list')

    def __init__(self, obj: Mapping[K, V]) -> None:
        self._keys = {k: i for i, k in enumerate(obj)}
        self._list = SharedList(obj.values())

    def __getitem__(self, key: K) -> V:
        idx = self._keys[key]
        return self._list[idx]

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(len={len(self)})'


def _serialize(v) -> Tensor:
    buf = pickle.dumps(v, protocol=-1)
    return torch.from_numpy(np.frombuffer(buf, dtype='u1').copy())


def _deserialize(x: Tensor):
    return pickle.loads(memoryview(x.numpy()))
