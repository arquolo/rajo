__all__ = ['Confusion', 'auc', 't_balance', 't_youden_j', 'youden_j']

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Staged, to_prob_sparse

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

    def __call__(self, pred: Tensor, true: Tensor) -> Tensor:
        c, pred, true = to_prob_sparse(pred, true)
        assert c == 2  # todo: support c == 1

        nt = self.bins + 1
        if not true.numel():
            return true.new_zeros(nt, 2, 2)

        # N/P support
        hist = true.bincount(minlength=2)

        # (N) of [0 .. max bin]
        pred = pred[:, 1].clamp(0, 1).mul_(self.bins).long()

        # (T 2) of FP, TP
        fp_tp = (pred * 2).add_(true).bincount(minlength=nt * 2).view(nt, 2)
        fp_tp = fp_tp.flipud().cumsum_(dim=0).flipud()

        # Endpoints
        fp_tp[0] = hist
        fp_tp[-1] = 0

        mat = torch.stack([hist - fp_tp, fp_tp], -1)  # (T 2 *2*)
        if not self.normalize:
            return mat
        return mat.float() / mat.sum((1, 2), keepdim=True)

    def collect(self, mat: Tensor) -> dict[str, Tensor]:
        return {'cm2t': mat, **super().collect(mat)}


def fpr_tpr(mat: Tensor) -> tuple[Tensor, Tensor]:
    """Tx2x2 matrix to pair of T-vectors"""
    assert mat.ndim == 3
    assert mat.shape[1:] == (2, 2)
    fpr, tpr = (mat[:, :, 1] / mat.sum(2).clamp_(_EPS)).unbind(1)
    return fpr, tpr


def auc(mat: Tensor) -> Tensor:
    """
    Area Under Receiver-Observer Curve.
    Tx2x2 matrix to scalar.
    """
    fpr, tpr = fpr_tpr(mat)
    return -torch.trapezoid(tpr, fpr)


def t_balance(mat: Tensor) -> Tensor:
    """
    Probability value where `sensitivity = specificity`.
    Tx2x2 matrix to scalar.
    """
    fpr, tpr = fpr_tpr(mat)
    idx = (tpr + fpr - 1).abs_().argmin()
    return idx.float() / (mat.shape[0] - 1)


def t_sup(mat: Tensor) -> Tensor:
    """
    Probability value where `P = PP` and `N = PN`, i.e. `FP = FN`.
    Tx2x2 matrix to scalar.
    """
    assert mat.shape[1:] == (2, 2)
    distance = F.mse_loss(mat.sum(1), mat.sum(2), reduction='none').sum(1)
    idx = distance.argmin()
    return idx.float() / (mat.shape[0] - 1)


def t_acc(mat: Tensor) -> Tensor:
    """
    Probability value where `TP + TN = max`.
    Tx2x2 matrix to scalar.
    """
    assert mat.shape[1:] == (2, 2)
    acc = (mat.diagonal(dim1=1, dim2=2).sum(1) / mat.sum((1, 2)).clamp_(_EPS))
    idx = acc.argmax()
    return idx.float() / (mat.shape[0] - 1)


def youden_j(mat: Tensor) -> Tensor:
    """
    Max of Youden's J statistic (i.e. informedness).
    Computed as `max(tpr - fpr)`, or `max(2 * balanced accuracy - 1)`.
    Tx2x2 matrix to scalar.
    """
    fpr, tpr = fpr_tpr(mat)
    return (tpr - fpr).amax()


def t_youden_j(mat: Tensor) -> Tensor:
    """
    Probability value where is max of Youden's J statistic.
    Tx2x2 matrix to scalar.
    """
    fpr, tpr = fpr_tpr(mat)
    idx = (tpr - fpr).argmax()
    return idx.float() / (mat.shape[0] - 1)


def support(mat: Tensor) -> Tensor:
    """CxC matrix to C-vector"""
    return mat[0].sum(1)
