__all__ = ['MultiheadProb', 'Prob']

from collections.abc import Sequence
from typing import Final

import torch
from torch import nn


class Prob(nn.Module):
    binary: Final[bool | None]

    def __init__(self, binary: bool | None = None):
        super().__init__()
        self.binary = binary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary = x.shape[1] == 1 if self.binary is None else self.binary
        return x.sigmoid() if binary else x.softmax(dim=1)


class MultiheadProb(nn.Module):
    heads: Final[list[int]]
    split: Final[bool]

    def __init__(self,
                 heads: Sequence[int],
                 binary: bool | None = None,
                 split: bool = False) -> None:
        super().__init__()
        self.heads = [*heads]
        self.split = split
        self.prob = Prob(binary=binary)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor] | torch.Tensor:
        heads = x.split(self.heads, dim=1)
        heads = [self.prob(h) for h in heads]
        if self.split:
            return heads
        return torch.stack(heads, dim=1)
