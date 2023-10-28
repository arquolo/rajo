__all__ = [
    'Confusion', 'auc', 't_balance', 't_dice', 't_otsu', 't_youden_j',
    'youden_j'
]

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Staged
from .func import roc_confusion

_EPS = torch.finfo(torch.float).eps


class Confusion(Staged):
    """Computes (T 2 2) confusion matrix, used for ROC and PR-based metrics"""
    def __init__(self,
                 bins: int = 64,
                 normalize: bool = True,
                 **funcs: Callable[[Tensor], Tensor]):
        super().__init__(**funcs)
        self.bins = bins
        self.normalize = normalize

    def __call__(self, y_pred: Tensor, y: Tensor, /) -> Tensor:
        mat = roc_confusion(y_pred, y)  # (T 2 *2*)
        if not self.normalize:
            return mat
        return mat.float() / mat.sum((1, 2), keepdim=True)

    def collect(self, mat: Tensor) -> dict[str, Tensor]:
        return {'cm2t': mat, **super().collect(mat)}


def fpr_tpr(mat: Tensor) -> tuple[Tensor, Tensor]:
    """Tx2x2 tensor to pair of T-vectors"""
    assert mat.ndim == 3
    assert mat.shape[1:] == (2, 2)
    fpr, tpr = (mat[:, :, 1] / mat.sum(2).clamp_min_(_EPS)).unbind(1)
    return fpr, tpr


def auc(mat: Tensor) -> Tensor:
    """
    Area Under Receiver-Observer Curve.
    Tx2x2 tensor to scalar.
    """
    fpr, tpr = fpr_tpr(mat)
    return -torch.trapezoid(tpr, fpr)


def t_balance(mat: Tensor) -> Tensor:
    """
    Probability value where `sensitivity = specificity`.
    Tx2x2 tensor to scalar.
    """
    fpr, tpr = fpr_tpr(mat)
    idx = (tpr + fpr - 1).abs_().argmin()
    return idx.float() / (mat.shape[0] - 1)


def t_sup(mat: Tensor) -> Tensor:
    """
    Probability value where `P = PP` and `N = PN`, i.e. `FP = FN`.
    Tx2x2 tensor to scalar.
    """
    assert mat.shape[1:] == (2, 2)
    distance = F.mse_loss(mat.sum(1), mat.sum(2), reduction='none').sum(1)
    idx = distance.argmin()
    return idx.float() / (mat.shape[0] - 1)


def t_acc(mat: Tensor) -> Tensor:
    """
    Probability value where `TP + TN = max`.
    Tx2x2 tensor to scalar.
    """
    assert mat.shape[1:] == (2, 2)
    acc = mat.diagonal(dim1=1, dim2=2).sum(1)
    acc = acc / mat.sum((1, 2)).clamp_min_(_EPS)
    idx = acc.argmax()
    return idx.float() / (mat.shape[0] - 1)


def youden_j(mat: Tensor) -> Tensor:
    """
    Max of Youden's J statistic (i.e. informedness).
    Computed as `max(tpr - fpr)`, or `max(2 * balanced accuracy - 1)`.
    Tx2x2 tensor to scalar.
    """
    fpr, tpr = fpr_tpr(mat)
    return (tpr - fpr).amax()


def t_youden_j(mat: Tensor) -> Tensor:
    """
    Probability value where is max of Youden's J statistic.
    Tx2x2 tensor to scalar.
    """
    fpr, tpr = fpr_tpr(mat)
    idx = (tpr - fpr).argmax()
    return idx.float() / (mat.shape[0] - 1)


def t_dice(mat: Tensor) -> Tensor:
    """
    Probability threshold for Dice score maximum.
    Tx2x2 tensor to scalar.
    """
    tp = mat[:, 1, 1]
    fp_tp = mat[:, :, 1].sum(1)
    fn_tp = mat[:, 1, :].sum(1)
    idx = (tp / (fp_tp + fn_tp).clamp_min_(_EPS)).argmax()
    return idx.float() / (mat.shape[0] - 1)


def support(mat: Tensor) -> Tensor:
    """CxC matrix to C-vector"""
    return mat[0].sum(1)


def t_otsu(mat: Tensor) -> Tensor:
    """
    Threshold for Otsu binarization. Minimizes weighted within-class variance
    of predicted probabilities.
    Tx2x2 tensor to scalar.
    """
    n = mat.shape[0]
    u = torch.arange(n - 1, device=mat.device).float().add_(0.5) / (n - 1)

    cdf = mat[:, :, 0].sum(1)  # (n)
    hist = torch.diff(cdf)  # (n - 1)

    x_cdf = F.pad((hist * u).cumsum(dim=0), [1, 0])  # (n)
    p_mean = x_cdf[-1]

    t = (x_cdf.square().div_(cdf.clamp_min(_EPS)) +
         (p_mean - x_cdf).square_().div_((1 - cdf).clamp_min_(_EPS))).argmax()
    return t / (n - 1)
