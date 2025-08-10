__all__ = ['rsqrt']

import math
from functools import partial
from typing import overload

from torch.optim import Optimizer, lr_scheduler


@overload
def rsqrt(
    optimizer: Optimizer, *, total_steps: int, warmup: int = ...
) -> lr_scheduler.LRScheduler: ...


@overload
def rsqrt(
    optimizer: Optimizer,
    *,
    epochs: int,
    steps_per_epoch: int,
    warmup: int = ...,
) -> lr_scheduler.LRScheduler: ...


def rsqrt(
    optimizer: Optimizer,
    *,
    total_steps: int | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    warmup: int = 100,
) -> lr_scheduler.LRScheduler:
    if total_steps is None:
        if epochs is None or steps_per_epoch is None:
            raise ValueError(
                'You must define either total_steps OR '
                '(epochs AND steps_per_epoch)'
            )
        assert epochs > 0
        assert steps_per_epoch > 0
        total_steps = epochs * steps_per_epoch
    assert total_steps > 0

    return lr_scheduler.LambdaLR(
        optimizer, partial(_get_rsqrt_lr, warmup, total_steps)
    )


def _get_rsqrt_lr(step: int, total: int, i: int) -> float:
    # i = 0 .. n-1
    i1 = i + 1
    warmup = i1 / step
    decay = math.sqrt(step / i1)
    cooldown = max(total - i1, 0) / math.sqrt((total - step) * step)
    return min(warmup, decay, cooldown)
