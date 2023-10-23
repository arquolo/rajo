__all__ = [
    'BCEWithLogitsLoss', 'CrossEntropyLoss', 'DiceLoss', 'LossWeighted',
    'MultiheadLoss'
]

from collections.abc import Sequence
from typing import Final

import torch
from torch import Tensor, nn

from rajo.distributed import reduce_if_needed

from .. import functional as F


class _Weighted(nn.Module):
    weight: Tensor | None
    reduce: Final[bool]

    def __init__(self,
                 num: int,
                 weight: Sequence[float] | Tensor | None = None,
                 reduce: bool = True) -> None:
        super().__init__()
        if weight is not None:
            if len(weight) == num:
                weight = torch.as_tensor(weight, dtype=torch.float)
                weight *= len(weight) / weight.sum()
            else:
                raise ValueError('each head should have weight')

        self.register_buffer('weight', weight)
        self.reduce = reduce

    def extra_repr(self) -> str:
        if self.weight is None:
            return ''
        return f'weight={self.weight.cpu().numpy().round(3)}'

    def _to_output(self, tensors: list[Tensor]) -> list[Tensor] | Tensor:
        if self.weight is not None:
            tensors = [t * w for t, w in zip(tensors, self.weight.unbind())]
        if not self.reduce:
            return tensors
        return torch.stack(torch.broadcast_tensors(*tensors), -1).mean(-1)


class MultiheadLoss(_Weighted):
    """
    Applies loss to each part of input.

    Parameters:
    - head_dims: list of C1, ..., Cn

    Argument shapes:
    - outputs: `(B, C1 + ... + Cn, ...)`,
    - targets: `(B, N, ...)` or same as outputs
    """
    head_dims: Final[list[int]]

    def __init__(
        self,
        base_loss: nn.Module,
        head_dims: Sequence[int],
        weight: Sequence[float] | Tensor | None = None,
        reduce: bool = True,
    ):
        super().__init__(len(head_dims), weight=weight, reduce=reduce)
        self.base_loss = base_loss
        self.head_dims = [*head_dims]
        self.num_heads = len(head_dims)

    def extra_repr(self) -> str:
        line = f'heads={self.head_dims}'
        if s := super().extra_repr():
            line += f', {s}'
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
    """Scales crossentropy loss w.r.t total sample size.

    `torch.nn.CrossEntropyLoss` scales loss by count of non-ignored samples,
    and if there're 0 of them, returns NAN.
    This one never returns NAN.

    If `full_size` is set, all samples are treated equally,
    and crossentropy is scaled by total sample size instead
    of count of non-ignored samples.

    If `full_size` is not set falls back to `torch.nn.CrossEntropyLoss` but
    properly weights samples across whole world.
    """
    full_size: Final[bool]

    def __init__(self,
                 weight: Tensor | None = None,
                 ignore_index: int = -100,
                 label_smoothing: float = 0,
                 full_size: bool = False) -> None:
        super().__init__(
            weight,
            ignore_index=ignore_index,
            reduction='mean',
            label_smoothing=label_smoothing)
        self.full_size = full_size

    def extra_repr(self) -> str:
        return 'full_size=True' if self.full_size else ''

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        loss = super().forward(outputs, targets)

        # Don't NAN
        # NOTE: support computed for local rank only to scale loss properly
        num_classes = outputs.shape[1]
        support = ((targets >= 0) &
                   (targets < num_classes)).mean(dtype=outputs.dtype)
        loss = torch.where(support > 0, loss, loss.new_zeros(loss.shape))

        if self.full_size:
            # Scale to be crossentropy(y_pred, y).sum() / y.size
            scale = support
        else:
            # Rescale to weight all samples equally across whole world
            w_support, = reduce_if_needed(support, mean=True)
            scale = torch.where(w_support > 0, support / w_support,
                                support.new_zeros(support.shape))

        return scale * loss


class DiceLoss(nn.Module):
    full_size: Final[bool]
    log: Final[bool]

    def __init__(self, full_size: bool = False, log: bool = False):
        super().__init__()
        self.full_size = full_size
        self.log = log

    def extra_repr(self) -> str:
        parts = []
        if self.full_size:
            parts += ['full_size=True']
        if self.log:
            parts += ['log=True']
        return ', '.join(parts)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return F.dice_loss(
            inputs, targets, full_size=self.full_size, log=self.log)
