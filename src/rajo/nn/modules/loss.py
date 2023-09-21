__all__ = [
    'CrossEntropyLoss', 'LossWeighted', 'MultiheadLoss',
    'NoisyBCEWithLogitsLoss'
]

from collections.abc import Sequence
from typing import Final

import torch
from torch import Tensor, nn


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


class NoisyBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    label_smoothing: Final[float]

    def __init__(self,
                 weight: Tensor | None = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 pos_weight: Tensor | None = None,
                 label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, reduce, reduction, pos_weight)
        self.label_smoothing = label_smoothing

    def extra_repr(self) -> str:
        return f'label_smoothing={self.label_smoothing}'

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        if outputs.requires_grad and (ls := self.label_smoothing):
            targets_ = torch.empty_like(targets)
            targets_.uniform_(-ls, ls).add_(targets).clamp_(0, 1)
        else:
            targets_ = targets
        return super().forward(outputs, targets)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Scales crossentropy loss w.r.t total sample size.

    Standard crossentropy scales loss by count of non-ignored samples,
    and if there're 0 of them, returns NAN.
    This one never returns NAN.

    If `full_size` is set, all samples are treated equally,
    and crossentropy is scaled by total sample size instead
    of count of non-ignored samples.
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
        num_classes = outputs.shape[1]
        support = ((targets >= 0) &
                   (targets < num_classes)).mean(dtype=outputs.dtype)
        loss = torch.where(support > 0, loss, loss.new_zeros(loss.shape))

        # Scale to be crossentropy(y_pred, y).sum() / y.size
        return (support * loss) if self.full_size else loss
