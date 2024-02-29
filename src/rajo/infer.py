__all__ = ['Indexer', 'MultiheadIndexer']

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt


class Indexer:
    rank: int
    labels: Mapping[str, npt.NDArray[np.int32]]

    def __init__(self, labels: Sequence[str]) -> None:
        self.rank = len(labels)
        self.labels = {t: np.array([i], 'i4') for i, t in enumerate(labels)}

    def label_indices(
        self,
        scores: npt.NDArray[np.number],
    ) -> npt.NDArray[np.int64]:
        """Get index (*) of most probable class from class scores (* C)."""
        return scores.argmax(-1)


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
    def __init__(self, labels: Mapping[str, Sequence[Sequence[int]]],
                 heads: Sequence[int]):
        nheads = len(heads)
        self._heads = heads
        *self._splits, self._total = np.cumsum(heads).tolist()

        self.labels = {}
        for t, sets in labels.items():
            lut = np.zeros(heads, np.bool_)
            for multi in sets:
                assert len(multi) <= nheads, \
                    f'Index {multi} is deeper than head count ({nheads})'
                loc = *(j if j != -1 else slice(None) for j in multi),
                lut[loc] = True
            self.labels[t] = np.argwhere(lut.ravel()).ravel().astype('i4')

        self.rank = int(np.prod(self._heads))

    def label_indices(
        self,
        scores: npt.NDArray[np.number],
    ) -> npt.NDArray[np.int64]:
        """Get index (*) of most probable class from class scores (* C)."""
        assert scores.shape[-1] == self._total, \
            f'Expected {self._total} channels, got {scores.shape[-1]}'

        multi = [h.argmax(-1) for h in np.split(scores, self._splits, axis=-1)]
        return np.ravel_multi_index(multi, self._heads)  # type: ignore
