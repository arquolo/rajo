__all__ = ['Indexer']

from collections.abc import Iterable, Mapping

import numpy as np


class Indexer:
    def __init__(self,
                 heads: Iterable[int] = (),
                 labels: Mapping[str, list[list[int]]] | Iterable[str] = ()):
        self.heads = [*heads]
        self.splits = []
        self.lut = None
        if not isinstance(labels, Mapping):
            return

        self.splits = np.cumsum(self.heads)[:-1].tolist()

        lut = np.full(self.heads, -1, dtype='i4')
        for i, sets in enumerate(labels.values()):
            for multi in sets:
                lut[tuple(j if j != -1 else slice(None) for j in multi)] = i
        if self.heads and (lut < 0).any():
            raise ValueError('Set of targets is incomplete')
        self.lut = lut.ravel()

    def label_indices(self, probs: np.ndarray) -> np.ndarray:
        if self.lut is None:
            return probs.argmax(-1)

        multi = [h.argmax(-1) for h in np.split(probs, self.splits, axis=-1)]
        idx = np.ravel_multi_index(multi, self.heads)
        return self.lut[idx]
