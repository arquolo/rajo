from collections.abc import Iterable

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor, einsum

# --------------------------------- set ops ----------------------------------


def _uniq(a: Tensor) -> set[int]:
    return set(torch.unique(a.cpu()).numpy())


def _is_subset(a: Tensor, sub: Iterable[int]) -> bool:
    return _uniq(a).issubset(sub)


def _is_unit_normalized(t: Tensor, dim: int = 1) -> bool:
    sum_ = t.sum(dim).float()
    return torch.allclose(sum_, torch.ones_like(sum_))


def _is_onehot(t: Tensor, dim: int = 1) -> bool:
    return _is_unit_normalized(t, dim) and _is_subset(t, [0, 1])


# -------------------------------- edt ---------------------------------------


# TODO: use pure torch impl
def torch_edt(mask: Tensor, sampling: tuple[float, ...] | None) -> Tensor:
    npy = mask.detach().cpu().numpy()

    npy_ = distance_transform_edt(npy, sampling)
    assert isinstance(npy_, np.ndarray)

    npy = npy_.astype('f')
    return torch.from_numpy(npy).to(device=mask.device)


# ------------------------------ onehot --------------------------------------


def index_to_onehot(indices: Tensor, num_classes: int) -> Tensor:
    """(b *) ints -> (b c *) uint8"""
    assert not indices.dtype.is_complex
    assert not indices.dtype.is_floating_point
    assert 0 <= indices.min() < num_classes, _uniq(indices)
    assert 0 <= indices.max() < num_classes, _uniq(indices)

    b, *star = indices.shape

    return torch.zeros(
        (b, num_classes, *star),
        dtype=torch.uint8,
        device=indices.device,
    ).scatter_(1, indices[:, None, ...], 1)


def onehot_to_distance(
    bchw: Tensor,
    sampling: tuple[float, ...] | None = None,
    dtype=None,
) -> Tensor:
    """(b c *) onehot probs -> (b c *) distances"""
    assert _is_onehot(bchw, dim=1)
    ret = torch.zeros_like(bchw, dtype=dtype)

    # TODO: remove loop to make GPU-friendly
    for b, chw in enumerate(bchw.bool().unbind()):
        for c, pos in enumerate(chw.unbind()):  # pos is (h w) of bools
            if pos.any():
                neg = ~pos
                neg_distance = torch_edt(neg, sampling)
                pos_distance = torch_edt(pos, sampling)
                ret[b, c] = neg_distance * neg - (pos_distance - 1) * pos
            # The idea is to leave blank the negative classes
            # since this is one-hot encoded,
            # another class will supervise that pixel

    return ret


def onehot_to_hd_distances(
    bchw: Tensor,
    sampling: tuple[float, ...] | None = None,
    dtype=None,
) -> Tensor:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap

    (b c *) onehot probs -> (b c *) distances
    """
    assert _is_onehot(bchw, dim=1)

    # TODO: remove loop to make GPU-friendly
    ret = torch.zeros_like(bchw, dtype=dtype)
    for b, chw in enumerate(bchw.bool().unbind()):
        for c, pos in enumerate(chw.unbind()):  # pos is (h w) of bools
            if pos.any():
                ret[b, c] = torch_edt(pos, sampling)

    return ret


# ------------------------------ losses -----------------------------------


class CrossEntropy:
    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        """(b c h w) probs, (b c h w) onehot -> ()"""
        assert probs.dtype.is_floating_point
        assert target.dtype.is_floating_point

        log_p = (probs + self.eps).log()
        loss = -einsum('bchw,bchw->', target, log_p)
        loss /= target.sum() + self.eps
        return loss


class GeneralizedDice:
    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        """(b c h w) (b c h w) -> ()"""
        assert probs.dtype.is_floating_point
        assert target.dtype.is_floating_point
        assert _is_unit_normalized(probs)
        assert _is_unit_normalized(target)

        # (b c)
        bc_target = einsum('bchw->bc', target)
        bc_probs = einsum('bchw->bc', probs)
        w = 1 / (bc_target + self.eps).square()

        if False:
            # (b c)
            intersection = w * einsum('bchw,bchw->bc', probs, target)
            union = w * (bc_probs + bc_target)

            # (b)
            num = einsum('bc->b', intersection) + self.eps
            den = einsum('bc->b', union) + self.eps
        else:
            # (b)
            num = einsum('bchw,bchw,bc->b', probs, target, w) + self.eps
            den = einsum('bc,bc->b', bc_probs + bc_target, w) + self.eps

        return 1 - 2 * (num / den).mean()


class DiceLoss:
    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        """(b c h w) (b c h w) -> ()"""
        assert probs.dtype.is_floating_point
        assert target.dtype.is_floating_point
        assert _is_unit_normalized(probs)
        assert _is_unit_normalized(target)

        intersection = einsum('bchw,bchw->bc', probs, target)
        union = einsum('bchw->bc', probs) + einsum('bchw->bc', target)

        num = torch.ones_like(intersection) - (2 * intersection + self.eps)
        den = union + self.eps
        return (num / den).mean()


class SurfaceLoss:
    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        """(b c h w) probs, (b c h w) distances -> ()"""
        assert probs.dtype.is_floating_point
        assert dist_maps.dtype.is_floating_point
        assert _is_unit_normalized(probs)
        assert not _is_onehot(dist_maps)

        return einsum('bchw,bchw->', probs, dist_maps) / probs.numel()


BoundaryLoss = SurfaceLoss


class HausdorffLoss:
    """
    Implementation heavily inspired from
    https://github.com/JunMa11/SegWithDistMap
    """

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        """(b c h w) probs, (b c h w) onehot -> ()"""
        assert probs.dtype.is_floating_point
        assert target.dtype.is_floating_point
        assert _is_unit_normalized(probs)
        assert _is_unit_normalized(target)
        assert probs.shape == target.shape

        tdm = onehot_to_hd_distances(target)

        zerotemp = index_to_onehot(probs.argmax(1), probs.shape[1])
        pdm = onehot_to_hd_distances(zerotemp)

        delta = (probs - target) ** 2
        dtm = tdm**2 + pdm**2
        return einsum('bchw,bchw->', delta, dtm) / delta.numel()


class FocalLoss:
    def __init__(self, gamma: float, eps: float = 1e-10) -> None:
        self.gamma = gamma
        self.eps = eps

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        """(b c h w) probs, (b c h w) onehot -> ()"""
        assert probs.dtype.is_floating_point
        assert target.dtype.is_floating_point
        assert _is_unit_normalized(probs)
        assert _is_unit_normalized(target)

        log_p = (probs + self.eps).log()
        w = (1 - probs) ** self.gamma

        loss = -einsum('bchw,bchw,bchw->', w, target, log_p)
        loss /= target.sum() + self.eps
        return loss
