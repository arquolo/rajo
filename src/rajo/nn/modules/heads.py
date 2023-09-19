__all__ = ['MultiheadAdapter', 'MultiheadProb', 'Prob']

from collections.abc import Iterable
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

    def __init__(self,
                 heads: Iterable[int],
                 binary: bool | None = None,
                 split: bool = False) -> None:
        super().__init__()
        self.heads = [*heads]
        self.split = split
        self.prob = Prob(binary=binary)

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        heads = x.split(self.heads, dim=1)
        heads = [self.prob(h) for h in heads]
        if self.split:
            return heads
        return torch.stack(heads, dim=1)


class MultiheadAdapter(nn.Module):
    weight: Tensor
    head_dims: Final[list[int]]
    eps: Final[float]
    prenorm: Final[bool]
    logits: Final[bool]

    def __init__(self,
                 c: int,
                 heads: Iterable[Iterable[Iterable[int]]],
                 eps: float = 1e-7,
                 prenorm: bool = True,
                 logits: bool = False) -> None:
        super().__init__()

        heads_ = [[[*cs] for cs in head] for head in heads]
        self.head_dims = [len(head) for head in heads_]

        total_labels = sum(self.head_dims)
        weight = torch.zeros(total_labels, c)
        for row, cs in zip(weight.unbind(),
                           (cs for head in heads_ for cs in head)):
            for c_ in cs:
                row[c_] = 1
        self.register_buffer('weight', weight)

        self.eps = eps
        self.prenorm = prenorm
        self.logits = logits

    def forward(self, x: Tensor) -> Tensor:
        if self.prenorm:
            x = x.softmax(dim=1)
        x = _linear_nd(x, self.weight)

        if self.logits:  # Raw logits
            return x.clamp_min(self.eps).log()

        # Per-head normalized probs
        return torch.cat(
            [h / h.sum(1, keepdim=True) for h in x.split(self.head_dims, 1)],
            dim=1,
        )


def _linear_nd(x: Tensor,
               weight: Tensor,
               bias: Tensor | None = None) -> Tensor:
    assert x.shape[1] == weight.shape[1]
    assert bias is None or bias.shape[0] == weight.shape[0]

    b, c, *volume = x.shape
    x = x.view(b, c, -1)
    x = F.conv1d(x, weight[:, :, None], bias)
    return x.view(b, -1, *volume)
