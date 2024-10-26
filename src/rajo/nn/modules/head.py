__all__ = ['MultiheadAdapter', 'MultiheadMaxAdapter', 'MultiheadProb', 'Prob']

from collections.abc import Iterable, Sequence
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Prob(nn.Module):
    binary: Final[bool | None]

    def __init__(self, binary: bool | None = None):
        super().__init__()
        self.binary = binary

    def forward(self, x: Tensor) -> Tensor:
        binary = x.shape[1] == 1 if self.binary is None else self.binary
        return x.sigmoid() if binary else x.softmax(dim=1)


class MultiheadProb(nn.Module):
    heads: Final[list[int]]
    split: Final[bool]

    def __init__(
        self,
        heads: Iterable[int],
        binary: bool | None = None,
        split: bool = False,
    ) -> None:
        super().__init__()
        self.heads = [*heads]
        self.split = split
        self.prob = Prob(binary=binary)

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        heads = x.split(self.heads, dim=1)
        heads = [self.prob(h) for h in heads]
        if self.split:
            return heads
        return torch.cat(heads, dim=1)


class MultiheadAdapter(nn.Module):
    weight: Tensor
    head_dims: Final[list[int]]
    eps: Final[float]
    from_logits: Final[bool]

    def __init__(
        self,
        c: int,
        heads: Sequence[Sequence[Iterable[int]]],
        eps: float = 1e-7,
        from_logits: bool = False,
    ) -> None:
        super().__init__()
        self.head_dims = [len(head) for head in heads]

        total_labels = sum(self.head_dims)
        weight = torch.zeros(total_labels, c)
        for row, cs in zip(
            weight.unbind(), (cs for head in heads for cs in head)
        ):
            for c_ in cs:
                row[c_] = 1
        self.register_buffer('weight', weight)

        self.eps = eps
        self.from_logits = from_logits

    def forward(self, x: Tensor) -> Tensor:
        if not self.from_logits:
            x = x.softmax(dim=1)
        x = _linear_nd(x, self.weight)

        if self.from_logits:  # Preserve logits
            return x.clamp(self.eps, 1 - self.eps).log()

        # Per-head normalized probs
        return torch.cat(
            [h / h.sum(1, keepdim=True) for h in x.split(self.head_dims, 1)],
            dim=1,
        )


class _SubsetMax(nn.Module):
    ids: Tensor | None
    default: Final[float]

    def __init__(self, ids: Sequence[int]) -> None:
        super().__init__()
        self.register_buffer('ids', torch.as_tensor(ids) if ids else None)
        self.default = float('-inf')

    def forward(self, x: Tensor) -> Tensor:
        if self.ids is not None:
            return x[:, self.ids].amax(dim=1)

        b, _, *volume = x.shape
        return x.new_full((b, *volume), fill_value=self.default)


class MultiheadMaxAdapter(nn.ModuleList):
    def __init__(self, heads: Iterable[Iterable[Sequence[int]]]) -> None:
        super().__init__([_SubsetMax(f) for head in heads for f in head])

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([m(x) for m in self], dim=1)


def _linear_nd(
    x: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    assert x.shape[1] == weight.shape[1]
    assert bias is None or bias.shape[0] == weight.shape[0]

    b, c, *volume = x.shape
    x = x.view(b, c, -1)
    x = F.conv1d(x, weight[:, :, None], bias)
    return x.view(b, -1, *volume)
