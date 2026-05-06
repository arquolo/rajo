__all__ = [
    'MultiheadAdapter',
    'MultiheadMaxAdapter',
    'MultiheadProb',
    'MultiheadSoftmax',
    'Prob',
]

from collections.abc import Iterable, Sequence
from functools import cache
from typing import Final, Literal

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
    """
    Parameters:
    - in_channels
    - heads - H sequences (one per each head)
      of S[h] sequences (one per each head's channel)
      of K[h,i] indices of new classes to build from (h,i) channel of input
    """

    nheads: Final[int]
    heads: Final[list[int]]
    c2h: Tensor  # input channel to head ID
    weight: Tensor
    eps: Final[float]
    from_logits: Final[bool]

    def __init__(
        self,
        in_channels: int,
        heads: Sequence[Sequence[Sequence[int]]],
        eps: float = 1e-7,
        from_logits: bool = False,
    ) -> None:
        super().__init__()
        self.nheads = len(heads)

        head_sizes = [len(head) for head in heads]
        self.heads = head_sizes

        index = _unpack_groups(head_sizes)
        self.register_buffer('index', index)

        output_labels = [cs for head in heads for cs in head]
        weight = torch.zeros(len(output_labels), in_channels)
        for row, out_labels in zip(weight.unbind(), output_labels):
            row[out_labels] = 1
        self.register_buffer('weight', weight)

        self.eps = eps
        self.from_logits = from_logits

    def forward(self, x: Tensor) -> Tensor:
        if self.from_logits:
            x = x.softmax(dim=1)
            x = _linear_nd(x, self.weight)
            return x.clamp_min(self.eps).log()

        x = _linear_nd(x, self.weight)

        # Per-head normalized probs
        sums = _reduce_groups(1, x, self.c2h, self.nheads, 'sum')
        normalized = [
            h / h_sum[:, None, ...]
            for h, h_sum in zip(
                x.split(self.heads, 1), sums.unbind(1), strict=True
            )
        ]
        return torch.cat(normalized, dim=1)


class _SubsetMax(nn.Module):
    """Maximum over selected channels, like `input[:, ids, ...].max(dim=1)`"""

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
    """Maximum over multiple series of selected channels.

    Accepts series of selected channels,
    each selection will get its own channel in output.
    """

    def __init__(self, ids_seq: Iterable[Sequence[int]]) -> None:
        super().__init__([_SubsetMax(ids) for ids in ids_seq])

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


class MultiheadSoftmax(nn.Module):
    """
    Convert logits to probabilities of set classes for multihead models.

    Parameters:
    - lut - ND-cube of targets labels, with shape equal to head sizes.
    - dim - dimension matching heads/classes.

    Requires `input.shape[dim] == sum(lut.shape)`
    """

    dim: Final[int]
    nheads: Final[int]
    from_log: Final[bool]
    eps: Final[float]
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
            c2h = _unpack_groups(heads)
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
            maxes = _reduce_groups(self.dim, x, c2h, self.nheads, 'max')
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


# -------------------------------- utilities ---------------------------------


def _unpack_groups(groups: Sequence[int]) -> Tensor:
    counts = torch.as_tensor(groups)
    return torch.arange(counts.shape[0]).repeat_interleave(counts)


def _reduce_groups(
    dim: int,
    x: Tensor,
    index: Tensor,
    ngroups: int,
    reduction: Literal['sum', 'prod', 'max'],
) -> Tensor:
    ishape = [1] * x.ndim
    ishape[dim] = -1
    index = index.view(ishape).expand_as(x)

    shape = list(x.shape)
    shape[dim] = ngroups
    if reduction == 'sum':
        ret = x.new_zeros(shape)
        if _allow_inplace(None, x):
            return ret.scatter_add_(dim, index, x)
        return ret.scatter_add(dim, index, x)

    ret = (
        x.new_ones(shape)
        if reduction == 'prod'
        else x.new_full(shape, float('-inf'))
    )
    if _allow_inplace(None, x):
        return ret.scatter_reduce_(dim, index, x, reduction)
    return ret.scatter_reduce(dim, index, x, reduction)


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


def _allow_inplace(mod: nn.Module | None, *xs: Tensor) -> bool:
    return not (
        (mod is None or mod.training)
        and torch.is_grad_enabled()
        and any(x.requires_grad for x in xs)
    )


@cache
def _make_eq(ndim: int, dim: int) -> str:
    if ndim > 15:
        raise ValueError(f'Input is too deep: {ndim}')
    axes = 'abcdefghijklmno'[:ndim]
    return f'{axes.replace(axes[dim], "q")},pq->{axes.replace(axes[dim], "p")}'
