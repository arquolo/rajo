"""Metrics computed without direct computation of confusion matrix"""

__all__ = ['accuracy', 'auroc', 'average_precision', 'dice_']

from typing import Literal

import torch
from torch import Tensor

from .func import class_ids, class_probs


def accuracy(y_pred: Tensor, y: Tensor, /) -> Tensor:
    # TODO: Add docs
    _, _, y_pred, y = class_ids(y_pred, y)
    return (y == y_pred).double().mean()


def _flat_dice(c: int, y_pred: Tensor, y: Tensor, /) -> Tensor:
    """Pair of N-vectors of int to C-vector."""
    assert y_pred.ndim == y.ndim == 1
    assert y_pred.shape == y.shape
    # true positive, positive & predicted positive counts
    tp, p, pp = (
        x.bincount(minlength=c).clamp_min_(1).double()
        for x in (y[y == y_pred], y, y_pred)
    )
    return 2 * tp / (p + pp)


def dice_(
    y_pred: Tensor,
    y: Tensor,
    /,
    *,
    mode: Literal['batchwise', 'imagewise'] = 'batchwise',
) -> Tensor:
    """Compute Dice metric for each class, result is C-vector.

    In `batchwise` mode Dice values are computed over flattened sample.
    Result cannot be averaged over epochs and suitable only
    as current batch statistics.

    In `imagewise` mode Dice values are computed per each sample in batch,
    then averaged.
    Such vectors CAN be averaged across all the batches, as they're
    sample-linear.
    """
    if mode == 'batchwise':
        _, c, y_pred, y = class_ids(y_pred, y)  # (m), (m)
        return _flat_dice(c, y_pred, y)

    b, c, y_pred, y = class_ids(y_pred, y, split_samples=True)  # (m), (m)
    value = _flat_dice(b * c, y_pred, y).view(b, c)
    return value.mean(dim=0)


def _rankdata(ten: Tensor) -> Tensor:
    sorter = ten.argsort()
    ten = ten[sorter]

    diff = torch.cat([ten.new_tensor([True]), ten[1:] != ten[:-1]])
    # diff = np.r_[True, ten[1:] != ten[:-1]]

    dense = diff.cumsum(0)[sorter.argsort()]

    diff = diff.nonzero(as_tuple=False).view(-1)
    count = torch.cat([diff, diff.new_tensor([diff.numel()])])
    # count = np.r_[diff.nonzero(diff).view(-1), diff.numel()]

    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _binary_metric(fn):
    """Applies specified function only on probabilities of indexed class"""

    def call(y_pred: Tensor, y: Tensor, /, *, index: int = 0) -> Tensor:
        y_pred, y = class_probs(y_pred, y)

        yc_pred = (
            (y_pred if index == 1 else 1 - y_pred)
            if y_pred.ndim == 1
            else y_pred[:, index]
        )
        yc = y == index

        return fn(yc_pred.view(-1), yc.view(-1))

    return call


@_binary_metric
def auroc(y_pred: Tensor, y: Tensor, /) -> Tensor:
    n = y.numel()
    n_pos = y.sum()

    r = _rankdata(y_pred)
    total = n_pos * (n - n_pos)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) // 2) / float(total)


@_binary_metric
def average_precision(y_pred: Tensor, y: Tensor, /) -> Tensor:
    n = y.numel()
    n_pos = y.sum()

    y = y[y_pred.argsort()].flipud()
    weights = torch.arange(1, n + 1).float().reciprocal()
    precision = y.cumsum(0).float()
    return torch.einsum('i,i,i', y.float(), precision, weights) / n_pos
