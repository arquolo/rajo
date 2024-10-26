__all__ = ['Indexer', 'MultiheadIndexer']

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import numpy.typing as npt
from glow import apack


class Indexer:
    rank: int  # `label_indices()` outputs values in [0..rank] range
    labels: Mapping[str, npt.NDArray[np.integer]]

    def __init__(self, labels: Sequence[str]) -> None:
        self.rank = len(labels)
        self.labels = {t: np.array([i]) for i, t in enumerate(labels)}
        self._dtype = np.min_scalar_type(self.rank - 1)

    def label_indices(
        self,
        scores: npt.NDArray[np.number],
    ) -> npt.NDArray[np.integer]:
        """Get index (*) of most probable class from class scores (* C)."""
        assert scores.shape[-1] == self.rank
        return scores.argmax(-1, out=np.empty(scores.shape[:-1], self._dtype))


class MultiheadIndexer(Indexer):
    """
    Parameters:
    - labels - mapping from label to set of indices (one per head).
      "-1" as index will disable matching head from calculation.
    - heads - count of sub-classes in each head.

    For example with such parameters:
    ```
    labels = {
        'A': [[0, -1]],
        'B': [[1, 0], [1, 1]],
        'C': [[1, 2]],
    }
    heads = [2, 3]
    ```
    Indexer will expect 5-channel scores (2 + 3) and will output
    "A" for [0, :] max, "B" for [1, 0] or [1, 1], and "C" for [1, 2].
    """

    lut: npt.NDArray[np.integer] | None = None

    def __init__(
        self,
        labels: Mapping[str, Sequence[Sequence[int]]],
        heads: Sequence[int],
        minimize: bool = True,
    ) -> None:
        num_heads = len(heads)
        self._total = sum(heads)

        vox = np.empty(heads, 'u1')  # 1-byte itemsize
        self.rank = vox.size  # Output range - [0..rank]
        dtype = np.min_scalar_type(vox.size - 1)

        self.labels = {}
        for t, sets in labels.items():
            lut = np.zeros(heads, np.bool_)
            for multi in sets:
                assert (
                    len(multi) <= num_heads
                ), f'Index {multi} is deeper than head count ({num_heads})'
                loc = tuple(j if j != -1 else slice(None) for j in multi)
                lut[loc] = True
            self.labels[t] = lut.ravel().nonzero()[0].astype(dtype)

        self._pnr = PackedNargmaxRavel(heads)
        if minimize:
            self._minimize()

    def _minimize(self) -> None:
        """Minimize range of output index using immediate LUT"""
        # Distribution of classes for each label
        mask = np.zeros((len(self.labels), self.rank), bool)
        for row, ids in zip(mask, self.labels.values()):
            row[ids] = True

        # Find unique cross-label usages of classes
        u, idx, lut = np.unique(
            mask, axis=1, return_index=True, return_inverse=True
        )
        lut = idx.argsort()[lut]  # Ensure ascending

        never_used = (u == 0).all(0)  # We have classes never used by any label
        if never_used.any():
            pos = never_used.argmax()
            tab = np.insert(np.arange(1, u.size), pos, 0)  # Unused becomes 0
            lut = tab[lut]
        else:
            lut += 1

        rank = int(lut.max() + 1)
        if rank == self.rank:  # Cannot minimize
            return
        lut = apack(lut)  # Compress LUT
        self.rank = rank
        self.lut = lut
        self.labels = {t: np.unique(lut[m]) for t, m in zip(self.labels, mask)}

    def label_indices(
        self,
        scores: npt.NDArray[np.number],
    ) -> npt.NDArray[np.integer]:
        """Get index (*) of most probable class from class scores (* C)."""
        assert (
            scores.shape[-1] == self._total
        ), f'Expected {self._total} channels, got {scores.shape[-1]}'

        r = self._pnr(scores)
        return self.lut[r] if self.lut is not None else r


class PackedNargmaxRavel:
    def __init__(self, heads: Sequence[int]) -> None:
        self._heads = heads
        self._num_heads = len(heads)

        # Head index for each channel
        self._idx = apack(np.arange(self._num_heads).repeat(heads))

        # Do uint8 optimization, 2x perf
        # Per-channel value offset
        self._idx256_u2 = (
            256 * (self._idx.astype('u2') + 1)
            if self._num_heads < 255
            else None
        )

        # First channel in head
        splits = np.cumsum([0, *heads[:-1]])
        self._splits = apack(splits)

        # Stride + bias vector
        strides = np.empty(heads, 'u1').strides  # 1-byte itemsize
        ddof = int(strides @ splits)
        mul = np.r_[strides, -ddof][None, :]
        self._scale = apack(mul)

    def __call__(
        self, scores: npt.NDArray[np.number]
    ) -> npt.NDArray[np.integer]:
        """Effectively the same as
        ```
            heads = np.split(scores, self._splits, axis=-1)
            multi = [h.argmax(-1) for h in heads]
            return np.ravel_multi_index(multi, self._heads)
        ```
        But 2-3x faster.
        """
        if self._num_heads == 1:
            return scores.argmax(-1, keepdims=True)

        *rshape, cc = scores.shape
        scores = scores.reshape(-1, cc)  # (n cc)

        # `argmax` chunks
        if self._idx256_u2 is not None and scores.dtype == 'u1':
            # (n cc)
            u2 = self._idx256_u2 - scores
            sortidx = cv2.sortIdx(u2, cv2.SORT_EVERY_ROW + cv2.SORT_ASCENDING)
            # (n h)
            r = sortidx[:, self._splits]
        else:
            # (n h)
            amax = np.maximum.reduceat(scores, self._splits, axis=1)
            _, cs = (amax.repeat(self._heads, axis=1) == scores).nonzero()
            h = self._idx[cs]
            i = np.r_[True, h[:-1] != h[1:]]  # use first maximum
            r = cs[i].reshape(scores.shape[0], self._num_heads)  # (n h)

        # Ravel index
        r = cv2.transform(r[None, :, :], self._scale)  # (n)
        return r.reshape(rshape)  # type: ignore
