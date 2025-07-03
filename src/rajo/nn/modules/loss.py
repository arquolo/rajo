__all__ = [
    'BCEWithLogitsLoss',
    'CrossEntropyLoss',
    'DiceLoss',
    'LossWeighted',
    'MultiheadLoss',
]

from collections.abc import Sequence
from typing import Final, Literal

import torch
from torch import Tensor, nn

from rajo.distributed import all_reduce

from .. import functional as F


class _Weighted(nn.Module):
    gain: Tensor | None
    reduction: Final[Literal['none', 'mean', 'sum']]

    def __init__(
        self,
        weight: Sequence[float] | Tensor | None = None,
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
    ) -> None:
        super().__init__()

        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(f'Unknown reduction mode: {reduction}')
        self.reduction = reduction

        if weight is not None:
            gain = torch.as_tensor(weight, dtype=torch.float)
            self.register_buffer('gain', gain.div_(gain.mean()))
        else:
            self.register_buffer('gain', None)

    def extra_repr(self) -> str:
        if self.gain is None:
            return ''
        return f'gain={self.gain.cpu().numpy().round(3)}'

    def _to_output(self, tensors: list[Tensor]) -> list[Tensor] | Tensor:
        if self.gain is not None:
            tensors = [t * w for t, w in zip(tensors, self.gain.unbind())]
        if self.reduction == 'none':
            return tensors

        t = torch.stack(torch.broadcast_tensors(*tensors), -1)
        t = F.finite_or_zero(t)
        return t.mean() if self.reduction == 'mean' else t.sum()


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
    channels: Final[int]
    renorm: Final[bool]
    unit_sum: Final[bool]

    def __init__(
        self,
        base_loss: nn.Module,
        head_dims: Sequence[int],
        weight: Sequence[float] | Tensor | None = None,
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
        renorm: bool = False,
        unit_sum: bool = True,
    ):
        self.head_dims = [*head_dims]
        self.num_heads = len(self.head_dims)
        self.channels = sum(self.head_dims)

        if weight is not None and self.num_heads != len(weight):
            raise ValueError(
                'Weight does not match head count. '
                f'{len(weight)} != {self.num_heads}'
            )

        super().__init__(weight, reduction=reduction)
        self.base_loss = base_loss
        self.renorm = renorm
        self.unit_sum = unit_sum

    def extra_repr(self) -> str:
        line = f'heads={self.head_dims}'
        if self.renorm:
            line = f'{line}, renorm=True'
        if not self.unit_sum:
            line = f'{line}, unit_sum=False'
        if s := super().extra_repr():
            line = f'{line}, {s}'
        return line

    def forward(
        self, outputs: Tensor, targets: Tensor
    ) -> Tensor | list[Tensor]:
        assert (
            outputs.shape[0] == targets.shape[0]
        ), 'outputs/targets differ in batch size'
        assert (
            outputs.shape[1] == self.channels
        ), 'output channel count does not match head dims'
        assert targets.shape[1] in (self.channels, self.num_heads), (
            'target channel count should match output, '
            'or be equal to head count'
        )
        assert (
            outputs.shape[2:] == targets.shape[2:]
        ), 'outputs/targets differ in sample size'

        o_parts = outputs.split(self.head_dims, dim=1)
        t_parts = (
            targets.unbind(dim=1)
            if targets.shape[1] == self.num_heads
            else targets.split(self.head_dims, dim=1)
        )

        tensors = [self.base_loss(o, t) for o, t in zip(o_parts, t_parts)]

        if self.renorm or not self.unit_sum:
            # Either renorm is ON or unit_sum is OFF (or both).
            # Get actual support from data
            sizes = [F.support(o, t) for o, t in zip(o_parts, t_parts)]
            support = torch.stack(sizes)
            (support,) = all_reduce(support, mean=True)

            if not self.renorm:  # Scale to world, not head size
                # Implies `unit_sum` also OFF.
                # Batch total gradient depends on count of non-ignored samples.
                # DDP-aware version of `sum` reduction.
                support = support.mean().broadcast_to(support.shape)

            if self.unit_sum:  # Normalize to unit sum, preserves grad norm
                # Implies `renorm` also ON.
                # Each batch will get the same total gradient.
                # DDP-aware version of `mean` reduction.
                support /= support.mean()

            # Scale each head
            tensors = [w * t for w, t in zip(support.unbind(), tensors)]

        return self._to_output(tensors)


class LossWeighted(_Weighted):
    def __init__(
        self,
        losses: Sequence[nn.Module],
        weight: Sequence[float] | None = None,
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
    ) -> None:
        if weight is not None and len(losses) != len(weight):
            raise ValueError(
                'Weight does not match loss count. '
                f'{len(weight)} != {len(losses)}'
            )

        super().__init__(weight, reduction=reduction)
        self.bases = nn.ModuleList(losses)

    def forward(
        self, outputs: Tensor, targets: Tensor
    ) -> Tensor | list[Tensor]:
        tensors = [m(outputs, targets) for m in self.bases]
        return self._to_output(tensors)


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    Drop-in replacement of `torch.nn.BCEWithLogitsLoss`
    with support of label smoothing.
    """

    smooth: Tensor | None
    end: Tensor

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
        pos_weight: Tensor | None = None,
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(weight, reduction=reduction, pos_weight=pos_weight)

        # Y(LS=0) -> Y, Y(LS=1) -> 1/2
        # Y <- lerp(Y, 1/2, weight=LS)
        self.register_buffer(
            'label_smoothing',
            (
                torch.as_tensor(label_smoothing, torch.float)
                if label_smoothing
                else None
            ),
        )
        self.register_buffer('end', torch.tensor(0.5))

    def extra_repr(self) -> str:
        return (
            ''
            if self.smooth is None
            else f'label_smoothing={self.smooth.item()}'
        )

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        # Target to float
        if not targets.dtype.is_floating_point:
            targets = targets.to(torch.get_default_dtype())

        # Smooth labels
        if self.smooth is not None:
            targets = targets.lerp(self.end, self.smooth)

        return super().forward(outputs, targets)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Drop-in replacement of `torch.nn.CrossEntropyLoss`.

    - returns 0 for empty batch (original gives NaN);
    - scales result to replica's sample size for even weight of samples
      (forces sample balance for DDP used with `ignore_index`).

    For global loss use `dist.all_reduce(loss, op=dist.ReduceOp.MEAN)`.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(
            weight,
            ignore_index=ignore_index,
            reduction='mean',
            label_smoothing=label_smoothing,
        )

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        loss = super().forward(outputs, targets)

        # NOTE: support is computed for current rank
        support = F.support(outputs, targets)
        (total_support,) = all_reduce(support, mean=True)
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
