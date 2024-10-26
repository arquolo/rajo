__all__ = [
    'class_ids',
    'class_probs',
    'confusion',
    'roc_confusion',
    'soft_confusion',
]

import torch
from einops import rearrange
from torch import Tensor

from rajo.distributed import all_reduce

_EPS = torch.finfo(torch.float).eps

# ---------------------------- prevalent classes -----------------------------


def confusion(
    y_pred: Tensor, y: Tensor, /, *, normalize: bool = False
) -> Tensor:
    """
    CxC confusion matrix over prevalent classes.

    Arguments:
    - y_pred - `(B C *)` or `(B 1 *)` logits;
    - y - `(B *)` indices.
    """
    _, c, y_pred, y = class_ids(y_pred, y)

    if not y.numel():
        return y.new_zeros(c, c)

    mat = (y * c).add_(y_pred).bincount(minlength=c * c).view(c, c)
    if not normalize:
        return mat
    return mat.float() / mat.sum().clamp_min_(1)


def class_ids(
    y_pred: Tensor,
    y: Tensor,
    /,
    *,
    split_samples: bool = False,
) -> tuple[int, int, Tensor, Tensor]:
    """
    Gets indices of prevalent classes.
    Keeps only valid labels (i.e. those in `0...C-1` range).

    If `split_samples` is used, then `divmod(result, c) = sample id, class id`.

    Arguments:
    - y_pred - `(B C *)` or `(B 1 *)` logits;
    - y - `(B *)` indices.

    Returns:
    - batch size;
    - number of classes;
    - N-vector of predicted indices;
    - N-vector of GT indices.
    """
    # y_pred (B C *), y (B *) - idx labels -> (B C *), (B *)
    if y_pred.ndim == y.ndim:
        y = y.squeeze(1)
    assert y_pred.shape[0] == y.shape[0], (y_pred.shape, y.shape)
    assert y_pred.shape[2:] == y.shape[1:], (y_pred.shape, y.shape)

    b, c = y_pred.shape[:2]
    if c == 1:
        y_pred = (y_pred > 0).squeeze(1).long()
        c = 2
    else:
        y_pred = y_pred.argmax(1)

    if y.dtype.is_floating_point:
        y = y.round().long()

    y_pred, y = y_pred.view(b, -1), y.view(b, -1)  # Flatten to [b *]
    m = (y >= 0) & (y < c)  # Mask to enforce valid range

    if split_samples:
        # Pack sample idx into class idx, such as
        #  n-th sample, c-th class gets `n * C + c` "class"
        db = torch.arange(0, b * c, c, dtype=y.dtype, device=y.device)[:, None]
        y_pred, y = y_pred + db, y + db  # [b *], [b *]

    y_pred, y = y_pred[m], y[m]  # Ensure GT range is ok

    return b, c, y_pred, y


# --------------------------- class probabilities ----------------------------


def soft_confusion(
    y_pred: Tensor, y: Tensor, /, *, normalize: bool = False
) -> Tensor:
    """
    CxC confusion matrix over class probabilities. World-wise.
    Preserves gradient.

    Arguments:
    - y_pred - `(B C *)` or `(B 1 *)` logits;
    - y - `(B *)` indices.

    Returns:
    - share of valid samples w.r.t. full batch;
    - CxC confusion matrix of floats.
    """
    y_pred, y = class_probs(y_pred, y)

    assert y_pred.dtype == torch.float32
    if y_pred.ndim == 2:  # multiclass (N C)
        c = y_pred.shape[1]
        mat = y_pred.new_zeros(c, c).index_add(0, y, y_pred)

    else:  # binary (N)
        pos = y_pred.new_zeros(2).index_add(0, y, y_pred)
        neg = y.bincount(minlength=2).float() - pos
        mat = torch.stack([neg, pos], 1)

    (mat,) = all_reduce(mat)
    if normalize:
        mat = mat / mat.sum().clamp_min(_EPS)
    return mat


def roc_confusion(
    y_pred: Tensor, y: Tensor, /, *, bins: int = 64, normalize: bool = False
) -> Tensor:
    """
    TxCxC confusion matrix over class probabilities computed
    for multiple thresholds. World-wise. Useful for AUROC, Youden J, AP.

    Arguments:
    - y_pred - `(B C *)` or `(B 1 *)` logits;
    - y - `(B *)` indices.
    """
    y_pred, y = class_probs(y_pred, y)

    nt = bins + 1
    if not y.numel():
        return y.new_zeros(nt, 2, 2)

    # N/P support
    hist = y.bincount(minlength=2)

    if y_pred.ndim == 2:  # multiclass (N 2)
        assert y_pred.shape[1] == 2, 'Non-binary prediction is not supported'
        y_pred = y_pred[:, 1]

    else:  # binary (N)
        assert y_pred.ndim == 1

    # (N) of [0 .. max bin]
    y_pred = y_pred.clamp(0, 1).mul_(bins).long()

    # (T 2) of FP, TP
    fp_tp = (y_pred * 2).add_(y).bincount(minlength=nt * 2).view(nt, 2)
    fp_tp = fp_tp.flipud().cumsum_(dim=0).flipud()

    # Endpoints
    fp_tp[0] = hist
    fp_tp[-1] = 0

    hist, fp_tp = all_reduce(hist, fp_tp)

    mat = torch.stack([hist - fp_tp, fp_tp], -1)  # (T 2 *2*)
    if not normalize:
        return mat
    return mat.float() / mat.sum((1, 2), keepdim=True)


def class_probs(y_pred: Tensor, y: Tensor, /) -> tuple[Tensor, Tensor]:
    """
    Gets probabilities of predicted classes.
    Keeps only valid labels (i.e. those in `0...C-1` range)

    Arguments:
    - y_pred - `(B C *)` or `(B 1 *)` logits;
    - y - `(B *)` indices.

    Returns:
    - number of classes;
    - share of valid samples w.r.t. full batch;
    - either NxC-matrix or N-vector of predicted probabilities;
    - N-vector of GT indices.
    """
    # y_pred (B C *), y (B *) - idx labels -> (B C *), (B *)
    # y_pred (B 1 *), y (B *) - binary -> (B 2 *), (B *)
    if y_pred.ndim == y.ndim:
        y = y.squeeze(1)
    assert y_pred.shape[0] == y.shape[0], (y_pred.shape, y.shape)
    assert y_pred.shape[2:] == y.shape[1:], (y_pred.shape, y.shape)

    if (c := y_pred.shape[1]) == 1:
        y_pred = y_pred.sigmoid().view(-1)  # (b n)
        c = 2
    else:
        y_pred = y_pred.softmax(dim=1)
        y_pred = rearrange(y_pred, 'b c ... -> (b ...) c')  # (b n) c

    if y.dtype.is_floating_point:
        y = y.round().long()
    y = y.ravel()  # (b n)

    # Ensure GT range is ok
    m = (y >= 0) & (y < c)
    y_pred, y = y_pred[m], y[m]  # y_pred: [(m) c] or (m), y: (m)

    return y_pred, y
