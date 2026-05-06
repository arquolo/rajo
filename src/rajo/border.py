from collections.abc import Iterable, Sequence

import torch
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
    pos: Tensor,
    sampling: Sequence[float] | None = None,
) -> Tensor:
    """(b c *) onehot probs -> (b c *) distances"""
    assert _is_onehot(pos, dim=1)

    # The idea is to leave blank the negative classes
    # since this is one-hot encoded,
    # another class will supervise that pixel
    neg = ~pos
    pos_distance = euclidean_distance_transform(pos, ndim=2, vx=sampling)
    neg_distance = euclidean_distance_transform(neg, ndim=2, vx=sampling)
    return neg_distance * neg + (1 - pos_distance) * pos


def onehot_to_hd_distances(
    bchw: Tensor,
    sampling: Sequence[float] | None = None,
) -> Tensor:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap

    (b c *) onehot probs -> (b c *) distances
    """
    assert _is_onehot(bchw, dim=1)
    return euclidean_distance_transform(bchw, ndim=2, vx=sampling)


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


# --------------------------- EDT -----------------------


def euclidean_distance_transform(
    x: Tensor, ndim: int | None = None, vx: Sequence[float] | None = None
) -> Tensor:
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor. Zeros will stay zero, and the distance will
        be propagated into nonzero voxels.
    ndim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x.masked_fill_(x > 0, float('inf'))
    ndim = ndim or x.ndim

    vx = [1] if vx is None else list(vx)
    if ndim is not None:
        vx = (vx + max(ndim - len(vx), 0) * [vx[-1]])[:ndim]

    if set(x.shape[-ndim:]) == {1}:  # Only 1s in shape
        return x

    if x.shape[-ndim] != 1:
        x = _l1dt_1d_(x.movedim(-ndim, 0), vx[0]).movedim(0, -ndim)
    if len(set(x.shape[-ndim + 1 :]) - {1}) <= 1:  # Only 1s in shape
        return x.abs()

    x.square_()
    for d, w in enumerate(vx[1:], -ndim + 1):
        if x.shape[d] != 1:
            x = _edt_1d(x.movedim(d, 0), d, w * w).movedim(0, d)
    return x.sqrt_()


@torch.jit.script
def _l1dt_1d_(f: Tensor, w: float = 1.0):
    """Algorithm 2 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    size = f.shape[0]
    for q in range(1, size):
        f[q] = torch.min(f[q], f[q - 1] + w)
    for q in range(size - 2, -1, -1):
        f[q] = torch.min(f[q], f[q + 1] + w)
    return f


@torch.jit.script
def _edt_1d(f: Tensor, w2: float = 1.0) -> Tensor:
    """Algorithm 1 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    f_len = f.shape[0]

    # rightmost parabola in lower envelope
    k = f.new_zeros(f.shape[1:], dtype=torch.long)

    # parabolas in lower envelope
    v = f.new_zeros(f.shape, dtype=torch.long)

    # boundaries between parabolas
    z = f.new_empty([f_len + 1, *f.shape[1:]])

    # compute lower envelope
    z[0] = float('-inf')
    z[1] = float('inf')
    for q in range(1, f_len):
        z, s = _edt_1d_intersection(f, v, z, k, q, w2)
        zk = z.gather(0, k[None])[0]
        mask = (k > 0) & (s <= zk)

        while mask.any():
            k.sub_(mask)

            z, s = _edt_1d_intersection(f, v, z, k, q, w2)
            zk = z.gather(0, k[None])[0]
            mask = (k > 0) & (s <= zk)

        s.masked_fill_(torch.isnan(s), float('-inf'))  # is this correct?

        k.add_(1)
        v.scatter_(0, k[None], q)
        z.scatter_(0, k[None], s[None])
        z.scatter_(0, k[None] + 1, float('inf'))

    # fill in values of distance transform
    k = f.new_zeros(f.shape[1:], dtype=torch.long)
    d = torch.empty_like(f)
    for q in range(f_len):
        zk = z.gather(0, k[None] + 1)[0]
        mask = zk < q

        while mask.any():
            k.add_(mask)
            zk = z.gather(0, k[None] + 1)[0]
            mask = zk < q

        vk = v.gather(0, k[None])[0]
        fvk = f.gather(0, vk[None])[0]
        d[q] = w2 * (q - vk).square() + fvk

    return d


@torch.jit.script
def _edt_1d_intersection(
    f: Tensor, v: Tensor, z: Tensor, k: Tensor, q: int, w2: float = 1.0
) -> tuple[Tensor, Tensor]:
    vk = v.gather(0, k[None])[0]
    fvk = f.gather(0, vk[None])[0]
    s = (f[q] - fvk) / (w2 * (q - vk)) + (q + vk)
    s = s / 2
    return z, s
