__all__ = [
    'Bias2d', 'BlurPool2d', 'Conv2dWs', 'Decimate2d', 'Laplace', 'Noise',
    'RgbToGray', 'Scale', 'Upscale2d'
]

from collections.abc import Iterable
from string import ascii_lowercase
from typing import Final

import cv2
import numpy as np
import torch
import torch.nn.functional as TF
from torch import nn

from .. import functional as F
from .util import to_buffers


class Scale(nn.Module):
    scale: Final[float]

    def __init__(self, scale: float = 255) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def extra_repr(self) -> str:
        return f'scale={self.scale}'


class Noise(nn.Module):
    std: Final[float]

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return torch.empty_like(x).normal_(std=self.std).add_(x)

    def extra_repr(self) -> str:
        return f'std={self.std}'


class Bias2d(nn.Module):
    def __init__(self,
                 dim: int,
                 *size: int,
                 device: torch.device | None = None):
        super().__init__()
        assert len(size) == 2
        self.bias = nn.Parameter(torch.empty(1, dim, *size, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.bias)

    def extra_repr(self) -> str:
        _, dim, *space = self.bias.shape
        return f'features={dim}, size={tuple(space)}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        size = [x.shape[2], x.shape[3]]

        if torch.jit.is_tracing() or bias.shape[2:] != size:
            # Stretch to input size
            bias = TF.interpolate(
                bias, size, mode='bicubic', align_corners=False)

        return x + bias


class Decimate2d(nn.Module):
    stride: Final[int]

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, ::self.stride, ::self.stride]

    def extra_repr(self) -> str:
        return f'stride={self.stride}'


class Upscale2d(nn.Module):
    """Upsamples input tensor in `scale` times.
    Use as inverse for `nn.Conv2d(kernel=3, stride=2)`.

    There're 2 different methods:

    - Pixels are thought as squares. Aligns the outer edges of the outermost
      pixels.
      Used in `torch.nn.Upsample(align_corners=True)`.

    - Pixels are thought as points. Aligns centers of the outermost pixels.
      Avoids the need to extrapolate sample values that are outside of any of
      the existing samples.
      In this mode doubling number of pixels doesn't exactly double size of the
      objects in the image.

    This module implements the second way (match centers).
    New image size will be computed as follows:
        `destination size = (source size - 1) * scale + 1`

    For comparison see [here](http://entropymine.com/imageworsener/matching).
    """
    stride: Final[int]

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.upscale2d(x, self.stride)

    def extra_repr(self):
        return f'stride={self.stride}'


class Conv2dWs(nn.Conv2d):
    """
    [Weight standartization](https://arxiv.org/pdf/1903.10520.pdf).
    Better use with GroupNorm(32, features).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d_ws(x, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)


# --------------------------------- blurpool ---------------------------------


def _pascal_triangle(n: int) -> list[int]:
    values = [1]
    for _ in range(n - 1):
        values = [a + b for a, b in zip([*values, 0], [0, *values])]
    return values[:n]


def _outer_mul(*ts: torch.Tensor) -> torch.Tensor:
    assert all(t.ndim == 1 for t in ts)
    letters = ascii_lowercase[:len(ts)]
    return torch.einsum(','.join(letters) + ' -> ' + letters, *ts)


class BlurPool2d(nn.Conv2d):
    def __init__(self,
                 dim: int,
                 kernel: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 padding_mode: str = 'reflect'):
        super().__init__(dim, dim, kernel, stride, padding, 1, dim, False,
                         padding_mode)
        to_buffers(self, persistent=False)

    def reset_parameters(self) -> None:
        if not self.in_channels:
            return

        weights = [
            torch.as_tensor(_pascal_triangle(k)).float()
            for k in self.kernel_size
        ]
        weight = _outer_mul(*weights)
        weight /= weight.sum()

        self.weight.copy_(weight, non_blocking=True)


# --------------------------------- laplace ----------------------------------


def _laplace_kernel(ksize: int, normalize: bool = True) -> torch.Tensor:
    assert ksize % 2 == 1, 'kernel must be odd'
    assert ksize <= 31, 'kernel must be not larger 31'

    ek = max(3, ksize)
    ep = ek // 2
    im = np.zeros((2 * ek - 1, 2 * ek - 1), 'f4')
    im[ek - 1, ek - 1] = 1

    scale = ksize / (4 ** ksize) if normalize else 1
    im = cv2.Laplacian(im, cv2.CV_32F, ksize=ksize, scale=scale)
    return torch.as_tensor(im[ep:ep + ek, ep:ep + ek])


class Laplace(nn.Conv2d):
    ksizes: Final[tuple[int, ...]]
    normalize: Final[bool]

    def __init__(self, ksizes: Iterable[int], normalize: bool = True):
        self.ksizes = *ksizes,
        self.normalize = normalize
        nk = len(self.ksizes)
        kmax = max(self.ksizes)
        super().__init__(1, nk, kmax, padding=kmax // 2, bias=False)
        to_buffers(self, persistent=False)

    def reset_parameters(self) -> None:
        kmax = max(max(self.ksizes), 3)  # noqa: PLW3301
        with torch.no_grad():
            self.weight.zero_()
        for k, sample in zip(self.ksizes, self.weight[:, 0, ...]):
            w = _laplace_kernel(k, normalize=self.normalize)
            p = (kmax - w.shape[0]) // 2
            with torch.no_grad():
                sample[p:kmax - p, p:kmax - p].copy_(w, non_blocking=True)

    def __repr__(self) -> str:
        nk = len(self.ksizes)
        return f'{type(self).__name__}(1, {nk}, kernel_sizes={self.ksizes})'


# -------------------------------- rgb 2 gray --------------------------------


class RgbToGray(nn.Conv2d):
    def __init__(self):
        super().__init__(3, 1, 1, bias=False)
        to_buffers(self, persistent=False)

    def reset_parameters(self) -> None:
        w = np.eye(3, dtype='f4')[None, :, :]  # (1 3 3)
        w = cv2.cvtColor(w, cv2.COLOR_RGB2GRAY)  # (1 3)
        w = w[:, :, None, None]  # (1 3 1 1)
        with torch.no_grad():
            self.weight.copy_(torch.from_numpy(w), non_blocking=True)
