__all__ = [
    'MultiheadAdapter',
    'MultiheadMaxAdapter',
    'MultiheadProb',
    'MultiheadSoftmax',
    'Prob',
]

from collections.abc import Iterable, Sequence
from functools import cache
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Prob(nn.Module):
    binary: Final[bool | None]

    def __init__(self, binary: bool | None = None) -> None:
        super().__init__()
        self.binary = binary

    def forward(self, x: Tensor) -> Tensor:
        binary = x.shape[1] == 1 if self.binary is None else self.binary
        return x.sigmoid() if binary else x.softmax(dim=1)


class MultiheadProb(nn.Module):
    """Semantically `cat(h.softmax() for h in input.split(heads))`"""

    heads: Final[list[int]]
    split: Final[bool]

    def __init__(
        self,
        heads: Iterable[int],
        binary: bool | None = None,
        split: bool = False,
    ) -> None:
        super().__init__()
        self.heads = [*heads]
        self.split = split
        self.prob = Prob(binary=binary)

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        heads = x.split(self.heads, dim=1)
        heads = [self.prob(h) for h in heads]
        if self.split:
            return heads
        return torch.cat(heads, dim=1)


class MultiheadAdapter(nn.Module):
    weight: Tensor
    head_dims: Final[list[int]]
    eps: Final[float]
    from_logits: Final[bool]

    def __init__(
        self,
        c: int,
        heads: Sequence[Sequence[Iterable[int]]],
        eps: float = 1e-7,
        from_logits: bool = False,
    ) -> None:
        super().__init__()
        self.head_dims = [len(head) for head in heads]

        total_labels = sum(self.head_dims)
        weight = torch.zeros(total_labels, c)
        for row, cs in zip(
            weight.unbind(), (cs for head in heads for cs in head)
        ):
            for c_ in cs:
                row[c_] = 1
        self.register_buffer('weight', weight)

        self.eps = eps
        self.from_logits = from_logits

    def forward(self, x: Tensor) -> Tensor:
        if not self.from_logits:
            x = x.softmax(dim=1)
        x = _linear_nd(x, self.weight)

        if self.from_logits:  # Preserve logits
            return x.clamp(self.eps, 1 - self.eps).log()

        # Per-head normalized probs
        return torch.cat(
            [h / h.sum(1, keepdim=True) for h in x.split(self.head_dims, 1)],
            dim=1,
        )


class _SubsetMax(nn.Module):
    ids: Tensor | None
    default: Final[float]

    def __init__(self, ids: Sequence[int]) -> None:
        super().__init__()
        self.register_buffer('ids', torch.as_tensor(ids) if ids else None)
        self.default = float('-inf')

    def forward(self, x: Tensor) -> Tensor:
        if self.ids is not None:
            return x[:, self.ids].amax(dim=1)

        b, _, *volume = x.shape
        return x.new_full((b, *volume), fill_value=self.default)


class MultiheadMaxAdapter(nn.ModuleList):
    def __init__(self, heads: Iterable[Iterable[Sequence[int]]]) -> None:
        super().__init__([_SubsetMax(f) for head in heads for f in head])

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([m(x) for m in self], dim=1)


def _linear_nd(
    x: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    assert x.shape[1] == weight.shape[1]
    assert bias is None or bias.shape[0] == weight.shape[0]

    b, c, *volume = x.shape
    x = x.view(b, c, -1)
    x = F.conv1d(x, weight[:, :, None], bias)
    return x.view(b, -1, *volume)


# ---------------------------- multihead softmax -----------------------------


class MultiheadSoftmax(nn.Module):
    """
    Convert logits to probabilities of set classes for multihead models.

    Parameters:
    - lut - ND-cube of targets labels, with shape equal to head sizes.
    - dim - dimension matching heads/classes.

    Requires `input.shape[dim] == sum(lut.shape)`
    """

    c2h: Tensor | None  # input channel to head ID, for maximum
    i2m: Tensor  # input channel to mid channel
    m2o: Tensor | None  # mid channel to output channel, optional

    def __init__(
        self,
        lut: Tensor,
        dim: int,
        from_log: bool = True,
        eps: float = 1e-3,
    ) -> None:
        out_channels = lut.max().item() + 1
        if lut.min().item() != 0 or lut.unique().numel() != out_channels:
            raise ValueError('LUT value range must be 0..out_channels-1')

        super().__init__()
        heads = lut.shape
        self.dim = dim
        self.nheads = len(heads)
        self.from_log = from_log
        self.eps = eps

        class_ids, locs = _factorize_lut(lut)
        locs = _make_absolute_offsets(locs, heads)

        in_channels = sum(heads)
        if locs.max().item() >= in_channels:  # Requires max
            in_channels += self.nheads
            c2h = (
                torch.arange(self.nheads)
                .repeat_interleave(torch.as_tensor(heads))
                .long()
            )
            self.register_buffer('c2h', c2h)
        else:
            self.c2h = None

        mid_channels = class_ids.shape[0]
        i2m = torch.zeros((mid_channels, in_channels)).scatter_(
            dim=1,
            index=locs,
            src=torch.as_tensor(1.0).expand(mid_channels, self.nheads),
        )
        self.register_buffer('i2m', i2m)

        self.out_channels = int(lut.max()) + 1
        if (  # Non final targets, requires conversion
            mid_channels != self.out_channels
            or (class_ids != torch.arange(self.out_channels)).any()
        ):
            self.register_buffer('m2o', class_ids)
        else:
            self.m2o = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.from_log:
            x = x.clamp_min(self.eps)
            x = x.log_() if _allow_inplace(self, x) else x.log()

        if (c2h := self.c2h) is not None:
            ishape = [1] * x.ndim
            ishape[self.dim] = -1
            h_index = c2h.view(ishape).expand_as(x)

            mshape = list(x.shape)
            mshape[self.dim] = self.nheads
            maxes = x.new_zeros(mshape)

            if _allow_inplace(self, x):
                maxes.scatter_reduce_(self.dim, h_index, x, 'max')
            else:
                maxes = maxes.scatter_reduce(self.dim, h_index, x, 'max')
            x = torch.cat([x, maxes], dim=self.dim)

        eq = _make_eq(x.ndim, self.dim)
        x = torch.einsum(eq, x, self.i2m)  # Logspace sum is probs multiply
        x = x.softmax(self.dim)

        if (m2o := self.m2o) is None:
            return x

        # Extract target labels
        oshape = list(x.shape)
        oshape[self.dim] = self.out_channels
        ret = x.new_zeros(oshape)

        # Sum probs to make true labels
        if _allow_inplace(self, x):
            ret.index_add_(self.dim, m2o, x)
        else:
            ret = ret.index_add(self.dim, m2o, x)
        return ret


def _factorize_lut(lut: Tensor) -> tuple[Tensor, Tensor]:
    """Convert ND lut to C sets of packed ND locs"""
    if not lut.size:
        return (
            lut.new_empty(0, dtype=torch.long),
            lut.new_empty((0, lut.ndim), dtype=torch.long),
        )

    grid = torch.meshgrid(*(torch.arange(s) for s in lut.shape), indexing='ij')
    locs = torch.stack([g.ravel() for g in grid], dim=-1)

    cls_locs = [
        (int(c), tuple[int, ...](loc))
        for c, loc in zip(lut.ravel().tolist(), locs.tolist())
    ]
    for i, hsize in enumerate(lut.shape):
        full_head = set(range(hsize))

        # Merge keys
        dups: dict[tuple[int, tuple[int, ...], tuple[int, ...]], set[int]] = {}
        for c, loc in cls_locs:
            dups.setdefault((c, loc[:i], loc[i + 1 :]), set()).add(loc[i])

        cls_locs = [
            (c, (*head, hc, *tail))
            for (c, head, tail), hcs in dups.items()
            for hc in ([-1] if hcs == full_head else sorted(hcs))
        ]

    cls_locs = sorted(cls_locs)
    return (
        lut.new_tensor([c for c, _ in cls_locs], dtype=torch.long),
        lut.new_tensor([loc for _, loc in cls_locs], dtype=torch.long),
    )


def _make_absolute_offsets(ids: Tensor, heads: Sequence[int]) -> Tensor:
    """Transform within-head IDs to absolute channel IDs"""
    offsets = torch.as_tensor([0, *heads]).cumsum_(dim=0)
    return torch.where(
        ids != -1,
        offsets[:-1] + ids,
        offsets[-1] + torch.arange(len(heads)),
    )


def _allow_inplace(mod: nn.Module, *xs: Tensor) -> bool:
    return not (
        mod.training
        and torch.is_grad_enabled()
        and any(x.requires_grad for x in xs)
    )


@cache
def _make_eq(ndim: int, dim: int) -> str:
    if ndim > 15:
        raise ValueError(f'Input is too deep: {ndim}')
    axes = 'abcdefghijklmno'[:ndim]
    return f'{axes.replace(axes[dim], "q")},pq->{axes.replace(axes[dim], "p")}'
