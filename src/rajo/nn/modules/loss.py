__all__ = ['LossBroadcast', 'LossWeighted', 'NoisyBCEWithLogitsLoss']

from collections.abc import Sequence
from typing import Final

import torch
from torch import nn


class LossBroadcast(nn.Module):
    """
    Applies loss to each part of input.

    Parameters:
    - head_dims: list of C1, ..., Cn

    Argument shapes:
    - outputs: `(B, C1 + ... + Cn, ...)`,
    - targets: `(B, N, ...)` or same as outputs
    """
    head_dims: Final[list[int]]
    weight: torch.Tensor | None
    reduce: Final[bool]

    def __init__(
        self,
        base_loss: nn.Module,
        head_dims: Sequence[int],
        head_weights: Sequence[float] | torch.Tensor | None = None,
        reduce: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.head_dims = [*head_dims]
        self.num_heads = len(head_dims)

        if head_weights is None:
            weight = None
        elif self.num_heads == len(head_weights):
            weight = torch.as_tensor(head_weights)
        else:
            raise ValueError('each head should have weight')

        self.register_buffer('weight', weight)
        self.reduce = reduce

    def extra_repr(self) -> str:
        line = f'heads={self.head_dims}'
        if self.weight is not None:
            line += f', weight={self.weight.tolist()}'
        return line

    def forward(self, outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        assert outputs.shape[0] == targets.shape[0]
        assert outputs.shape[1] == sum(self.head_dims)
        assert outputs.shape[2:] == targets.shape[2:]
        o_parts = outputs.split(self.head_dims, dim=1)
        t_parts = (
            targets.unbind(dim=1) if targets.shape[1] == self.num_heads else
            targets.split(self.head_dims, dim=1))

        tensors = [self.base_loss(o, t) for o, t in zip(o_parts, t_parts)]

        return _weight_sum_reduce(
            *tensors, weight=self.weight, reduce=self.reduce)


class LossWeighted(nn.Module):
    weight: torch.Tensor | None
    reduce: Final[bool]

    def __init__(self,
                 losses: Sequence[nn.Module],
                 weights: Sequence[float] | None = None,
                 reduce: bool = True) -> None:
        super().__init__()
        self.bases = nn.ModuleList(losses)

        if weights is None:
            self.register_buffer('weight', None)
        elif len(weights) == len(self.bases):
            self.register_buffer('weight', torch.as_tensor(weights))
        else:
            raise ValueError('each loss should have weight')

        self.reduce = reduce

    def extra_repr(self) -> str:
        if self.weight is None:
            return ''
        return f'weight={self.weight.tolist()}'

    def forward(self, outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        tensors = [m(outputs, targets) for m in self.bases]
        return _weight_sum_reduce(
            *tensors, weight=self.weight, reduce=self.reduce)


class NoisyBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    label_smoothing: Final[float]

    def __init__(self,
                 weight: torch.Tensor | None = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 pos_weight: torch.Tensor | None = None,
                 label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, reduce, reduction, pos_weight)
        self.label_smoothing = label_smoothing

    def extra_repr(self) -> str:
        return f'label_smoothing={self.label_smoothing}'

    def forward(self, outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        if outputs.requires_grad and (ls := self.label_smoothing):
            targets_ = torch.empty_like(targets)
            targets_.uniform_(-ls, ls).add_(targets).clamp_(0, 1)
        else:
            targets_ = targets
        return super().forward(outputs, targets)


def _weight_sum_reduce(
        *tensors: torch.Tensor,
        weight: torch.Tensor | None = None,
        reduce: bool = True) -> list[torch.Tensor] | torch.Tensor:
    tensors_ = list(tensors)
    if weight is not None:
        tensors_ = [t * w for t, w in zip(tensors_, weight.unbind())]
    if not reduce:
        return tensors_
    return torch.stack(torch.broadcast_tensors(*tensors_), -1).sum(-1)
