__all__ = ['Lambda', 'Metric', 'Scores', 'Staged', 'compose']

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Protocol, overload

from glow import coroutine
from torch import Tensor


@dataclass(frozen=True)
class Scores:
    scalars: dict[str, float | int] = field(default_factory=dict)
    tensors: dict[str, Tensor] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, mapping: dict[str, Tensor]) -> 'Scores':
        obj = cls()
        for k, v in mapping.items():
            if v.numel() == 1:
                obj.scalars[k] = v.item()
            else:
                obj.tensors[k] = v
        return obj


class MetricFn(Protocol):
    def __call__(self, pred, true) -> Tensor: ...


class Metric(ABC):
    """Base class for metric"""

    @abstractmethod
    def __call__(self, pred, true) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def collect(self, state: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError


class Lambda(Metric):
    """Wraps arbitary loss function to metric"""

    fn: MetricFn

    @overload
    def __init__(self, fn: Callable, name: str): ...

    @overload
    def __init__(self, fn: MetricFn, name: None = ...): ...

    def __init__(self, fn, name=None):
        self.fn = fn
        self.name = fn.__name__ if name is None else name

    def __call__(self, pred, true) -> Tensor:
        return self.fn(pred, true)

    def collect(self, state: Tensor) -> dict[str, Tensor]:
        return {self.name: state}


class Staged(Metric):
    """Makes metric a "producer": applies multiple functions to its "state" """

    def __init__(self, **funcs: Callable[[Tensor], Tensor]):
        self.funcs = funcs

    def collect(self, state: Tensor) -> dict[str, Tensor]:
        return {key: fn(state) for key, fn in self.funcs.items()}


@coroutine
def _batch_averaged(
    fn: Metric,
) -> Generator[dict[str, Tensor], Sequence[Tensor], None]:
    assert isinstance(fn, Metric)
    args = yield {}
    state = fn(*args)
    for n in count(2):
        args = yield fn.collect(state)
        state.lerp_(fn(*args), 1 / n)


@coroutine
def compose(*fns: Metric) -> Generator[Scores, Sequence[Tensor], None]:
    updates = tuple(_batch_averaged(fn) for fn in fns)
    args = yield Scores()
    while True:
        scores: dict[str, Tensor] = {}
        for u in updates:
            scores |= u.send(args)
        args = yield Scores.from_dict(scores)
