__all__ = ['conv2d_ws', 'dice_loss', 'outer_mul', 'upscale2d']

from string import ascii_lowercase

import torch
import torch.nn.functional as F
from torch import Size, Tensor, nn
from torch.nn.utils import parametrize

from rajo.metrics.confusion import dice
from rajo.metrics.func import soft_confusion

_EPS = torch.finfo(torch.float).eps

_size = Size | list[int] | tuple[int, ...]


# @torch.jit.script
def upscale2d(x: Tensor, stride: int = 2) -> Tensor:
    # ! stride-aware fallback, works everywhere
    # x = F.interpolate(x, None, self.stride)
    # return F.avg_pool2d(x, self.stride, 1, 0)

    # ! matches to single libtorch op, complex torchscript op
    pad = 1 - stride
    size = [x.shape[2], x.shape[3]]
    size = [s * stride + pad for s in size]
    return F.interpolate(x, size, mode='bilinear', align_corners=True)


def conv2d_ws(x: Tensor,
              weight: Tensor,
              bias: Tensor | None = None,
              stride: _size | int = 1,
              padding: _size | int | str = 0,
              dilation: _size | int = 1,
              groups: int = 1):
    weight = F.layer_norm(weight, weight.shape[1:], eps=1e-5)
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)


def standartize_conv_weights(model: nn.Module) -> None:
    """Enforce weight standartization for all ConvNd modules"""
    for m in model.modules():
        if not isinstance(m, nn.modules.conv._ConvNd):
            continue
        shape = m.weight.shape
        parametrize.register_parametrization(
            m, 'weight', nn.LayerNorm(shape[1:], elementwise_affine=False))


def outer_mul(*ts: Tensor) -> Tensor:
    """Outer product of series of 1D-tensors"""
    assert all(t.ndim == 1 for t in ts)
    letters = ascii_lowercase[:len(ts)]
    return torch.einsum(','.join(letters) + ' -> ' + letters, *ts)


def dice_loss(y_pred: Tensor,
              y: Tensor,
              /,
              *,
              full_size: bool = False,
              log: bool = False) -> Tensor:
    support, mat = soft_confusion(y_pred, y)

    mat = mat / mat.sum().clamp_min(_EPS)
    score = dice(mat)
    loss = -score.clamp_min(_EPS).log().mean() if log else (1 - score.mean())

    # 0 if empty
    loss = torch.where(
        loss.new_tensor(support > 0, dtype=torch.bool), loss,
        torch.zeros_like(loss))

    return (support * loss) if full_size else loss
