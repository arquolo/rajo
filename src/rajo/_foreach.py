__all__ = [
    'add_',
    'addcdiv_',
    'addcmul_',
    'clamp_max_',
    'div',
    'lerp',
    'lerp_',
    'maximum_',
    'mul',
    'mul_',
    'norm',
    'sqrt',
    'zero_',
]

from collections.abc import Iterable

import torch
from torch import Tensor, jit

type Number = int | float | bool


def add_(
    self: Iterable[Tensor],
    other: Iterable[Tensor] | Number,
    *,
    alpha: Number = 1,
) -> None:
    """`self += other * alpha`"""
    self = list(self)
    if isinstance(other, int | float | bool):
        assert alpha == 1
        if can_do_foreach(self):
            torch._foreach_add_(self, other)
        else:
            for s in self:
                s.add_(other)
    else:
        other = list(other)
        if can_do_foreach(self + other):
            torch._foreach_add_(self, other, alpha=alpha)
        else:
            for s, o in zip(self, other, strict=True):
                s.add_(o, alpha=alpha)


def addcdiv_(
    self: Iterable[Tensor],
    tensor1: Iterable[Tensor],
    tensor2: Iterable[Tensor],
    *,
    value: Number = 1,
) -> None:
    """`self += value * tensor1 / tensor2`"""
    self = list(self)
    tensor1 = list(tensor1)
    tensor2 = list(tensor2)
    if can_do_foreach(self + tensor1 + tensor2):
        torch._foreach_addcdiv_(self, tensor1, tensor2, value=value)
    else:
        for s, t1, t2 in zip(self, tensor1, tensor2):
            s.addcdiv_(t1, t2, value=value)


def addcmul_(
    self: Iterable[Tensor],
    tensor1: Iterable[Tensor],
    tensor2: Iterable[Tensor],
    *,
    value: Number = 1,
) -> None:
    """`self += value * tensor1 * tensor2`"""
    self = list(self)
    tensor1 = list(tensor1)
    tensor2 = list(tensor2)
    if can_do_foreach(self + tensor1 + tensor2):
        torch._foreach_addcmul_(self, tensor1, tensor2, value=value)
    else:
        for s, t1, t2 in zip(self, tensor1, tensor2):
            s.addcmul_(t1, t2, value=value)


def clamp_max_(
    self: Iterable[Tensor], other: Iterable[Tensor] | Number = 1
) -> None:
    """`self.clamp_()`"""
    self = list(self)
    if isinstance(other, Iterable):
        other = list(other)
        if can_do_foreach(self + other):
            torch._foreach_clamp_max_(self, other)
        else:
            for s, o in zip(self, other):
                s.clamp_max_(o)
    elif can_do_foreach(self):
        torch._foreach_clamp_max_(self, other)
    else:
        for s in self:
            s.clamp_max_(other)


def div(
    self: Iterable[Tensor], other: Iterable[Tensor] | Number = 1
) -> list[Tensor]:
    """`self / other`"""
    self = list(self)
    if isinstance(other, Iterable):
        other = list(other)
        if can_do_foreach(self + other):
            return list(torch._foreach_div(self, other))
        return [x / y for x, y in zip(self, other)]

    if can_do_foreach(self):
        return list(torch._foreach_div(self, other))
    return [x / other for x in self]


def lerp(
    self: Iterable[Tensor], other: Iterable[Tensor], *, weight: Number
) -> list[Tensor]:
    """`lerp(self, other, t)`"""
    self = list(self)
    other = list(other)
    if can_do_foreach(self + other):
        return list(torch._foreach_lerp(self, other, weight))
    return [s.lerp(o, weight) for s, o in zip(self, other)]


def lerp_(
    self: Iterable[Tensor], other: Iterable[Tensor], *, weight: Number
) -> None:
    """`self = lerp(self, other, t)` or `self += t * (other - self)`"""
    self = list(self)
    other = list(other)
    if can_do_foreach(self + other):
        torch._foreach_lerp_(self, other, weight)
    else:
        for s, o in zip(self, other):
            s.lerp_(o, weight)


def maximum_(self: Iterable[Tensor], other: Iterable[Tensor]) -> None:
    """`self = max(self, other)`"""
    self = list(self)
    other = list(other)
    if can_do_foreach(self + other):
        torch._foreach_maximum_(self, other)
    else:
        for s, o in zip(self, other):
            torch.max(s, o, out=s)


def mul(
    self: Iterable[Tensor], other: Iterable[Tensor] | Tensor | Number = 1
) -> list[Tensor]:
    """`self * other`"""
    self = list(self)
    if isinstance(other, Iterable):
        other = list(other)
        if can_do_foreach(self + other):
            return list(torch._foreach_mul(self, other))
        return [x * y for x, y in zip(self, other)]

    if can_do_foreach(self):
        return list(torch._foreach_mul(self, other))
    return [x * other for x in self]


def mul_(self: Iterable[Tensor], other: Iterable[Tensor] | Number = 1) -> None:
    """`self *= other`"""
    self = list(self)
    if isinstance(other, Iterable):
        other = list(other)
        if can_do_foreach(self + other):
            torch._foreach_mul_(self, other)
        else:
            for s, f in zip(self, other):
                s.mul_(f)

    elif can_do_foreach(self):
        torch._foreach_mul_(self, other)
    else:
        for s in self:
            s.mul_(other)


def norm(
    self: Iterable[Tensor], p: Number = 2, dtype: torch.dtype | None = None
) -> list[Tensor]:
    """`self.norm(p, dtype=)`"""
    self = list(self)
    if can_do_foreach(self):
        return list(torch._foreach_norm(self, p, dtype))
    return [s.norm(p, dtype=dtype) for s in self]


def sqrt(self: Iterable[Tensor]) -> list[Tensor]:
    """`sqrt(self)`"""
    self = list(self)
    if can_do_foreach(self):
        return list(torch._foreach_sqrt(self))
    return [torch.sqrt(s) for s in self]


def zero_(self: Iterable[Tensor]) -> None:
    """`self = 0`"""
    self = list(self)
    if can_do_foreach(self):
        torch._foreach_zero_(self)
    else:
        for s in self:
            s.zero_()


def can_do_foreach(tensors: Iterable[Tensor]) -> bool:
    if jit.is_scripting():
        return False
    return all(tsr is None or not tsr.is_sparse for tsr in tensors)
