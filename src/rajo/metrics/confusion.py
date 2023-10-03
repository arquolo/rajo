__all__ = [
    'Confusion',
    'SoftConfusion',
    'accuracy',
    'accuracy_',  # per-class
    'accuracy_balanced',  # mean of per-class
    'dice',  # mean of per-class
    'dice_',  # per-class
    'iou',
    'kappa',
    'kappa_quadratic_weighted',
    'sensitivity',
    'specificity',
]

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import nn

from .base import Staged, to_index_sparse, to_prob_sparse

_EPS = torch.finfo(torch.float).eps


class Confusion(Staged):
    """Confusion Matrix. Returns 2d tensor"""
    def __call__(self, pred: Tensor, true: Tensor) -> Tensor:
        c, pred, true = to_index_sparse(pred, true)

        if not true.numel():
            return true.new_zeros(c, c)

        mat = (true * c).add_(pred).bincount(minlength=c * c).view(c, c)
        return mat.float() / mat.sum()

    def collect(self, mat: Tensor) -> dict[str, Tensor]:
        c = mat.shape[0]
        return {f'cm{c}': mat, **super().collect(mat)}


class SoftConfusion(Confusion):
    """Confusion Matrix which can be used for loss functions"""
    def __call__(self, pred: Tensor, true: Tensor) -> Tensor:
        c, pred, true = to_prob_sparse(pred, true)

        assert pred.dtype == torch.float32
        mat = pred.new_zeros(c, c).index_add(0, true, pred)

        if dist.is_initialized() and dist.get_world_size() > 1:
            mat = nn.all_reduce(mat)
            assert mat is not None
        return mat / mat.sum()


def accuracy(mat: Tensor) -> Tensor:
    """CxC matrix to scalar"""
    return mat.trace() / mat.sum().clamp(_EPS)


def accuracy_(mat: Tensor) -> Tensor:
    """CxC matrix to C-vector"""
    return (mat.diag() / mat.sum(1).clamp(_EPS))


def specificity(mat: Tensor) -> Tensor:
    """2x2 matrix to scalar"""
    assert mat.shape == (2, 2)
    a0, _ = accuracy_(mat).unbind()
    return a0


def sensitivity(mat: Tensor) -> Tensor:
    """2x2 matrix to scalar"""
    assert mat.shape == (2, 2)
    _, a1 = accuracy_(mat).unbind()
    return a1


def accuracy_balanced(mat: Tensor) -> Tensor:
    """CxC matrix to scalar"""
    return accuracy_(mat).mean()


def kappa(mat: Tensor) -> Tensor:
    """CxC matrix to scalar"""
    expected = mat.sum(0) @ mat.sum(1)
    observed = mat.trace()
    return 1 - (1 - observed) / (1 - expected).clamp(_EPS)


def kappa_quadratic_weighted(mat: Tensor) -> Tensor:
    """CxC matrix to scalar"""
    assert mat.shape[0] == mat.shape[1]
    r = torch.arange(mat.shape[0], device=mat.device)

    weights = (r[:, None] - r[None, :]).float() ** 2
    weights /= weights.max()

    expected = mat.sum(0) @ weights @ mat.sum(1)
    observed = mat.view(-1) @ weights.view(-1)
    return 1 - observed / expected.clamp(_EPS)


def iou(mat: Tensor) -> Tensor:
    """CxC matrix to C-vector"""
    return mat.diag() / (mat.sum(0) + mat.sum(1) - mat.diag()).clamp(_EPS)


def dice_(mat: Tensor) -> Tensor:
    """CxC matrix to C-vector, full Dice score"""
    return 2 * mat.diag() / (mat.sum(0) + mat.sum(1)).clamp(_EPS)


def dice(mat: Tensor) -> Tensor:
    """CxC matrix to scalar"""
    return dice_(mat).mean()


def support(mat: Tensor) -> Tensor:
    """CxC matrix to C-vector"""
    return mat.sum(1)
