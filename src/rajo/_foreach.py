__all__ = [
    'add_', 'addcdiv_', 'addcmul_', 'lerp', 'lerp_', 'maximum_', 'mul_',
    'sqrt', 'zero_'
]

from collections.abc import Iterable

import torch
from torch import Tensor

Number = int | float | bool


def add_(self: Iterable[Tensor],
         other: Iterable[Tensor] | Number,
         *,
         alpha: Number = 1):
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


def addcdiv_(self: Iterable[Tensor],
             tensor1: Iterable[Tensor],
             tensor2: Iterable[Tensor],
             *,
             value: Number = 1):
    """`self += value * tensor1 / tensor2`"""
    self = list(self)
    tensor1 = list(tensor1)
    tensor2 = list(tensor2)
    if can_do_foreach(self + tensor1 + tensor2):
        torch._foreach_addcdiv_(self, tensor1, tensor2, value=value)
    else:
        for s, t1, t2 in zip(self, tensor1, tensor2):
            s.addcdiv_(t1, t2, value=value)


def addcmul_(self: Iterable[Tensor],
             tensor1: Iterable[Tensor],
             tensor2: Iterable[Tensor],
             *,
             value: Number = 1):
    """`self += value * tensor1 * tensor2`"""
    self = list(self)
    tensor1 = list(tensor1)
    tensor2 = list(tensor2)
    if can_do_foreach(self + tensor1 + tensor2):
        torch._foreach_addcmul_(self, tensor1, tensor2, value=value)
    else:
        for s, t1, t2 in zip(self, tensor1, tensor2):
            s.addcmul_(t1, t2, value=value)


def lerp(self: Iterable[Tensor], other: Iterable[Tensor], *,
         weight: Number) -> list[Tensor]:
    """`lerp(self, other, t)`"""
    self = list(self)
    other = list(other)
    if can_do_foreach(self + other):
        return torch._foreach_lerp(self, other, weight)
    return [s.lerp(o, weight) for s, o in zip(self, other)]


def lerp_(self: Iterable[Tensor], other: Iterable[Tensor], *, weight: Number):
    """`self = lerp(self, other, t)` or `self += t * (other - self)`"""
    self = list(self)
    other = list(other)
    if can_do_foreach(self + other):
        torch._foreach_lerp_(self, other, weight)
    else:
        for s, o in zip(self, other):
            s.lerp_(o, weight)


def maximum_(self: Iterable[Tensor], other: Iterable[Tensor]):
    """`self = max(self, other)`"""
    self = list(self)
    other = list(other)
    if can_do_foreach(self + other):
        torch._foreach_maximum_(self, other)
    else:
        for s, o in zip(self, other):
            torch.max(s, o, out=s)


def mul_(self: Iterable[Tensor], *, scalar: Number = 1):
    """`self *= scalar`"""
    self = list(self)
    if can_do_foreach(self):
        torch._foreach_mul_(self, scalar)
    else:
        for s in self:
            s.mul_(scalar)


def sqrt(self: Iterable[Tensor]) -> list[Tensor]:
    """`sqrt(self)`"""
    self = list(self)
    if can_do_foreach(self):
        return torch._foreach_sqrt(self)
    return [torch.sqrt(s) for s in self]


def zero_(self: Iterable[Tensor]):
    """`self = 0`"""
    self = list(self)
    if can_do_foreach(self):
        torch._foreach_zero_(self)
    else:
        for s in self:
            s.zero_()


def can_do_foreach(tensors: Iterable[Tensor]) -> bool:
    if torch.jit.is_scripting():
        return False
    return all(tsr is None or not tsr.is_sparse for tsr in tensors)
