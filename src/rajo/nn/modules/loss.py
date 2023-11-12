__all__ = [
    'BCEWithLogitsLoss', 'CrossEntropyLoss', 'DiceLoss', 'LossWeighted',
    'MultiheadLoss'
]

from collections.abc import Sequence
from typing import Final, Literal

import torch
from torch import Tensor, nn

from rajo.distributed import reduce_if_needed

from .. import functional as F


class _Weighted(nn.Module):
    gain: Tensor | None
    reduce: Final[bool]

    def __init__(self,
                 num: int,
                 weight: Sequence[float] | Tensor | None = None,
                 reduce: bool = True) -> None:
        super().__init__()
        if weight is not None:
            if len(weight) != num:
                raise ValueError('each head should have weight')
            gain = torch.as_tensor(weight, dtype=torch.float)
            gain *= len(gain) / gain.sum()
        else:
            gain = None

        self.register_buffer('gain', gain)
        self.reduce = reduce

    def extra_repr(self) -> str:
        if self.gain is None:
            return ''
        return f'gain={self.gain.cpu().numpy().round(3)}'

    def _to_output(self, tensors: list[Tensor]) -> list[Tensor] | Tensor:
        if self.gain is not None:
            tensors = [t * w for t, w in zip(tensors, self.gain.unbind())]
        tensors = [F.finite_or_zero(t) for t in tensors]
        if not self.reduce:
            return tensors
        return torch.stack(torch.broadcast_tensors(*tensors), -1).mean(-1)


class MultiheadLoss(_Weighted):
    """
    Applies loss to each part of input.

    Parameters:
    - head_dims: list of C1, ..., Cn
    - if renorm is set, each head loss is scaled to its sample size

    Argument shapes:
    - outputs: `(B, C1 + ... + Cn, ...)`,
    - targets: `(B, N, ...)` or same as outputs
    """
    head_dims: Final[list[int]]
    num_heads: Final[int]
    renorm: Final[bool]

    def __init__(
        self,
        base_loss: nn.Module,
        head_dims: Sequence[int],
        weight: Sequence[float] | Tensor | None = None,
        reduce: bool = True,
        renorm: bool | Literal['raw'] = False,
    ):
        super().__init__(len(head_dims), weight=weight, reduce=reduce)
        self.base_loss = base_loss
        self.head_dims = [*head_dims]
        self.num_heads = len(head_dims)
        self.renorm = renorm

    def extra_repr(self) -> str:
        line = f'heads={self.head_dims}'
        if self.renorm:
            line = f'{line}, renorm={self.renorm}'
        if s := super().extra_repr():
            line = f'{line}, {s}'
        return line

    def forward(self, outputs: Tensor,
                targets: Tensor) -> Tensor | list[Tensor]:
        assert outputs.shape[0] == targets.shape[0]
        assert outputs.shape[1] == sum(self.head_dims)
        assert outputs.shape[2:] == targets.shape[2:]
        o_parts = outputs.split(self.head_dims, dim=1)
        t_parts = (
            targets.unbind(dim=1) if targets.shape[1] == self.num_heads else
            targets.split(self.head_dims, dim=1))

        tensors = [self.base_loss(o, t) for o, t in zip(o_parts, t_parts)]

        if self.renorm:  # Scale each head to whole its size
            sizes = [F.support(o, t) for o, t in zip(o_parts, t_parts)]
            support = torch.stack(sizes)
            support, = reduce_if_needed(support, mean=True)

            if self.renorm != 'raw':  # Normalize to unit sum
                support /= support.mean()

            tensors = [w * t for w, t in zip(support.unbind(), tensors)]

        return self._to_output(tensors)


class LossWeighted(_Weighted):
    def __init__(self,
                 losses: Sequence[nn.Module],
                 weight: Sequence[float] | None = None,
                 reduce: bool = True) -> None:
        super().__init__(len(losses), weight=weight, reduce=reduce)
        self.bases = nn.ModuleList(losses)

    def forward(self, outputs: Tensor,
                targets: Tensor) -> Tensor | list[Tensor]:
        tensors = [m(outputs, targets) for m in self.bases]
        return self._to_output(tensors)


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    Drop-in replacement of `torch.nn.BCEWithLogitsLoss`
    with support of label smoothing.
    """
    label_smoothing: Final[float]

    def __init__(self,
                 weight: Tensor | None = None,
                 reduction: str = 'mean',
                 pos_weight: Tensor | None = None,
                 label_smoothing: float = 0) -> None:
        super().__init__(weight, reduction=reduction, pos_weight=pos_weight)
        self.label_smoothing = label_smoothing

    def extra_repr(self) -> str:
        return f'label_smoothing={self.label_smoothing}'

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        # Target to float
        if not targets.dtype.is_floating_point:
            targets = targets.to(torch.get_default_dtype())

        if smoothing := self.label_smoothing:
            delta = 1 - smoothing
            eps = smoothing / 2
            targets = (targets * delta).add_(eps)

        return super().forward(outputs, targets)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Drop-in replacement of `torch.nn.CrossEntropyLoss`.

    - returns 0 for empty batch (original gives NaN);
    - scales result to replica's sample size for even weight of samples
      (forces sample balance for DDP used with `ignore_index`).

    For global loss use `dist.all_reduce(loss, op=dist.ReduceOp.MEAN)`.
    """
    def __init__(self,
                 weight: Tensor | None = None,
                 ignore_index: int = -100,
                 label_smoothing: float = 0) -> None:
        super().__init__(
            weight,
            ignore_index=ignore_index,
            reduction='mean',
            label_smoothing=label_smoothing)

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        loss = super().forward(outputs, targets)

        # NOTE: support is computed for current rank
        support = F.support(outputs, targets)
        total_support, = reduce_if_needed(support, mean=True)
        if total_support is support:
            return loss

        # Rescale loss to weight all samples equally across whole world
        # Do not NAN on empty GT
        loss = loss * (support / total_support)
        return F.finite_or_zero(loss)


class DiceLoss(nn.Module):
    """DDP-aware Dice loss. Returns same value on all replicas"""
    log: Final[bool]

    def __init__(self, log: bool = False):
        super().__init__()
        self.log = log

    def extra_repr(self) -> str:
        return 'log=True' if self.log else ''

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return F.dice_loss(inputs, targets, log=self.log)
