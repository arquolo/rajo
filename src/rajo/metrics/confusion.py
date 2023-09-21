__all__ = [
    'Confusion', 'ConfusionGrad', 'accuracy', 'accuracy_balanced', 'iou',
    'kappa', 'kappa_quadratic_weighted'
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


class ConfusionGrad(Confusion):
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
    return mat.trace() / mat.sum().clamp(_EPS)


def accuracy_balanced(mat: Tensor) -> Tensor:
    return (mat.diag() / mat.sum(1).clamp(_EPS)).mean()


def kappa(mat: Tensor) -> Tensor:
    expected = mat.sum(0) @ mat.sum(1)
    observed = mat.trace()
    return 1 - (1 - observed) / (1 - expected).clamp(_EPS)


def kappa_quadratic_weighted(mat: Tensor) -> Tensor:
    assert mat.shape[0] == mat.shape[1]
    r = torch.arange(mat.shape[0], device=mat.device)

    weights = (r[:, None] - r[None, :]).float() ** 2
    weights /= weights.max()

    expected = mat.sum(0) @ weights @ mat.sum(1)
    observed = mat.view(-1) @ weights.view(-1)
    return 1 - observed / expected.clamp(_EPS)


def iou(mat: Tensor) -> Tensor:
    return mat.diag() / (mat.sum(0) + mat.sum(1) - mat.diag()).clamp(_EPS)


def dice(mat: Tensor) -> Tensor:
    return 2 * mat.diag() / (mat.sum(0) + mat.sum(1)).clamp(_EPS)


def dice_mean(mat: Tensor) -> Tensor:
    return dice(mat).mean()


def support(mat: Tensor) -> Tensor:
    return mat.sum(1)
