__all__ = [
    'detach_',
    'device',
    'dump_to_onnx',
    'eval_',
    'frozen',
    'inference',
    'param_count',
    'profile',
]

import functools
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from glow import si
from torch import Tensor, nn


def _apply[T](xs: T, fn: Callable[[Tensor], Any]) -> T:
    if isinstance(xs, Tensor):
        return fn(xs)

    if isinstance(xs, str | bytes | np.ndarray):
        return xs  # type: ignore

    if isinstance(xs, tuple) and hasattr(xs, '_fields'):  # namedtuple
        return type(xs)(*(_apply(x, fn) for x in xs))  # type: ignore
    if isinstance(xs, Mapping):
        return dict(_apply(kv, fn) for kv in xs.items())  # type: ignore
    if isinstance(xs, Iterable):
        return type(xs)(_apply(x, fn) for x in xs)  # type: ignore
    return xs


def device() -> torch.device:
    """Gets current device, including CPU"""
    return torch.device(
        f'cuda:{torch.cuda.current_device()}'
        if torch.cuda.is_available()
        else 'cpu'
    )


def param_count(module: nn.Module) -> int:
    """Count of parameters/buffers in net, both training and not"""
    tensors = set(module.parameters()) | set(module.buffers())
    return si(sum(t.numel() for t in tensors if not nn.parameter.is_lazy(t)))


@contextmanager
def eval_(module: nn.Module) -> Iterator[None]:
    """
    Switches all children to eval mode.
    Restores train/eval distribution at exit.
    """
    were_train = {m for m in module.modules() if m.training}
    try:
        module.eval()
        yield
    finally:
        for m in module.modules():
            if m in were_train:
                m.training = True  # Don't call .train() as it's recursive.


@contextmanager
def detach_(module: nn.Module) -> Iterator[None]:
    """Prevents module from changing its parameters.

    Forbids autograd to record operations on parameters in this module, thus
    excluding them from gradient computation.

    This method is helpful for freezing part of the module for finetuning or
    training parts of a model individually (e.g., GAN training).

    NEITHER disable gradient flow NOR prevents buffers to change.
    """
    required_grad = {
        p.detach_()
        for p in module.parameters()
        if not nn.parameter.is_lazy(p) and p.requires_grad
    }
    try:
        yield
    finally:
        for p in required_grad:
            p.requires_grad_()


@contextmanager
def frozen(module: nn.Module) -> Iterator[None]:
    """Blocks module from changing state of its parameters and buffers.

    Switches all children to eval mode and detaches all parameters.
    DOES NOT disable gradient flow.
    """
    with eval_(module), detach_(module):
        yield


@contextmanager
def inference(module: nn.Module) -> Iterator[None]:
    """Enables inference mode for module.

    Switches all children to eval mode.
    Disables gradient flow.

    All the tensors created in this mode are marked as inference,
    and they are NOT COMPATIBLE WITH AUTOGRAD AT ALL
    (used in JIT, backward, etc.).

    DON'T use this mode to initialize lazy modules.
    """
    with eval_(module), torch.inference_mode():
        yield


# ----------------------------- profile CUDA ops -----------------------------


def profile[**P, R](fn: Callable[P, Iterator[R]]) -> Callable[P, Iterator[R]]:
    """Decorator to profile CUDA ops. Use with `nvprof`

    Use in script launched via:
    ```bash
    nvprof --profile-from-start off -o trace.prof -- python main.py
    ```
    Usage:
    >>> @profile
    ... def train_loop():
    ...     for data in loader:
    ...         yield step(data)

    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[R]:
        results = fn(*args, **kwargs)
        with torch.cuda.profiler.profile():
            yield from islice(results, 1)
            with torch.autograd.profiler.emit_nvtx():
                yield from results

    return functools.update_wrapper(wrapper, fn)


def dump_to_onnx(
    filepath: Path | str,
    model: nn.Module,
    *shapes: tuple[int, ...],
    device: str | torch.device = 'cpu',
) -> None:
    """Converts model to ONNX graph

    Parameters:
    - model - torch.nn.Module to convert
    - shapes - Shapes of input data, all except batch dimension

    Example usage:
    >>> module = torch.nn.Linear(4, 4)
    >>> filename = 'model.onnx'
    >>> dump_to_onnx(filename, module, [4])

    To restore graph:
    >>> from onnxruntime import backend
    >>> rep = backend.prepare(filename, device='cpu')
    >>> rep.run([np.zeros(4, 4)])[0]

    """
    dynamic_axes = {
        f'inp_{i}': {
            0: 'batch',
            **{dim: f'inp_{i}_dim_{dim}' for dim in range(2, 1 + len(shape))},
        }
        for i, shape in enumerate(shapes)
    }
    torch.onnx.export(
        model.to(device).eval(),
        tuple(
            torch.rand(1, *s, requires_grad=True, device=device)
            for s in shapes
        ),
        filepath,
        opset_version=17,
        input_names=[*dynamic_axes],
        dynamic_axes=dynamic_axes,
    )
