__all__ = [
    'Bias2d',
    'BlurPool2d',
    'Conv2dWs',
    'Decimate2d',
    'Laplace',
    'Noise',
    'RgbToGray',
    'Scale',
    'Upscale2d',
]

from collections.abc import Iterable
from typing import Final

import torch
import torch.nn.functional as TF
from torch import Tensor, jit, nn

from .. import functional as F
from .. import ops
from .util import to_buffers


class Scale(nn.Module):
    scale: Final[float]

    def __init__(self, scale: float = 255) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale

    def extra_repr(self) -> str:
        return f'scale={self.scale}'


class Noise(nn.Module):
    std: Final[float]

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        return torch.empty_like(x).normal_(std=self.std).add_(x)

    def extra_repr(self) -> str:
        return f'std={self.std}'


class Bias2d(nn.Module):
    def __init__(
        self, dim: int, *size: int, device: torch.device | None = None
    ):
        super().__init__()
        assert len(size) == 2
        self.bias = nn.Parameter(torch.empty(1, dim, *size, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.bias)

    def extra_repr(self) -> str:
        _, dim, *space = self.bias.shape
        return f'features={dim}, size={tuple(space)}'

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias
        size = [x.shape[2], x.shape[3]]

        if jit.is_tracing() or bias.shape[2:] != size:
            # Stretch to input size
            bias = TF.interpolate(
                bias, size, mode='bicubic', align_corners=False
            )

        return x + bias


class Decimate2d(nn.Module):
    stride: Final[int]

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :: self.stride, :: self.stride]

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

    def forward(self, x: Tensor) -> Tensor:
        return F.upscale2d(x, self.stride)

    def extra_repr(self):
        return f'stride={self.stride}'


class Conv2dWs(nn.Conv2d):
    """
    [Weight standartization](https://arxiv.org/pdf/1903.10520.pdf).
    Better use with GroupNorm(32, features).
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d_ws(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# --------------------------------- blurpool ---------------------------------


class BlurPool2d(nn.Conv2d):
    def __init__(
        self,
        dim: int,
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        padding_mode: str = 'reflect',
    ):
        super().__init__(
            dim, dim, kernel, stride, padding, 1, dim, False, padding_mode
        )
        to_buffers(self, persistent=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if not self.in_channels:
            return

        weights = [ops.pascal_triangle(k).float() for k in self.kernel_size]
        weight = F.outer_mul(*weights)
        weight /= weight.sum()

        self.weight.copy_(weight, non_blocking=True)


# --------------------------------- laplace ----------------------------------


class Laplace(nn.Conv2d):
    ksizes: Final[tuple[int, ...]]
    normalize: Final[bool]

    def __init__(self, ksizes: Iterable[int], normalize: bool = True):
        self.ksizes = (*ksizes,)
        self.normalize = normalize
        nk = len(self.ksizes)
        kmax = max(self.ksizes)
        super().__init__(1, nk, kmax, padding=kmax // 2, bias=False)
        to_buffers(self, persistent=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        kernels = [
            ops.laplace_kernel(k, normalize=self.normalize)
            for k in self.ksizes
        ]
        # Do center padding
        kmax = max(max(self.ksizes), 3)  # noqa: PLW3301
        self.weight.zero_()
        for w, dst in zip(kernels, self.weight[:, 0, ...]):
            pad = (kmax - w.shape[0]) // 2
            dst[pad : kmax - pad, pad : kmax - pad].copy_(w, non_blocking=True)

    def __repr__(self) -> str:
        nk = len(self.ksizes)
        return f'{type(self).__name__}(1, {nk}, kernel_sizes={self.ksizes})'


# -------------------------------- rgb 2 gray --------------------------------


class RgbToGray(nn.Conv2d):
    def __init__(self):
        super().__init__(3, 1, 1, bias=False)
        to_buffers(self, persistent=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        w = ops.rgb2gray_kernel()[:, :, None, None]  # (1 3 1 1)
        self.weight.copy_(w, non_blocking=True)
