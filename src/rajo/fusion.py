__all__ = [
    'flatten',
    'flatten_seq',
    'fuse_conv_bn',
    'pad_conv_sym_same',
    'remove_infer_no_ops',
]
import math
from functools import partial, singledispatch
from typing import TYPE_CHECKING, TypeGuard

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models.resnet import BasicBlock, Bottleneck
from wrapt import register_post_import_hook

if TYPE_CHECKING:
    # timm~=1.0.26
    # segmentation-models-pytorch~=0.5.0
    from segmentation_models_pytorch.base.modules import Activation, Attention
    from segmentation_models_pytorch.decoders.fpn.decoder import (
        Conv3x3GNReLU,
        SegmentationBlock,
    )
    from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
    from timm.models import EfficientNetFeatures, MobileNetV3Features
    from timm.models._efficientnet_blocks import (
        ConvBnAct,
        DepthwiseSeparableConv,
        InvertedResidual,
    )

_SEQ_FWD = nn.Sequential.forward
# TODO: test if it's ok for convtranspose blocks
_CONV_TYPES: list[type] = [nn.Conv2d, nn.ConvTranspose2d]
_BN_TYPES: list[type] = [nn.BatchNorm2d]
_NO_OP_TYPES: tuple[type, ...] = (nn.Identity, nn.modules.dropout._DropoutNd)

# ------------------------------ kernel padding ------------------------------


@singledispatch
def pad_conv_sym_same(net: nn.Module) -> nn.Module:
    """Fixes checkerboard patterns in gradients"""
    children = {name: pad_conv_sym_same(m) for name, m in net.named_children()}
    for name, m in children.items():
        setattr(net, name, m)
    return net


@pad_conv_sym_same.register(nn.Conv2d)
@pad_conv_sym_same.register(nn.ConvTranspose2d)
def _pad_conv_sym_same(net: nn.Module) -> nn.Module:
    if not isinstance(net, nn.modules.conv._ConvNd):
        return net
    if net.stride == 1:
        return net

    if isinstance(net.padding, str):
        raise NotImplementedError(f'Unsupported padding: {net.padding}')

    # Only for input divisible by stride, "0" is for that
    eff_kernel = tuple(
        map(_get_effective_kernel, net.kernel_size, net.dilation)
    )
    full_padding = tuple(
        map(partial(_get_full_padding, 0), eff_kernel, net.stride)
    )
    # Check if padding is symmetric
    if all(hp * 2 == fp for hp, fp in zip(net.padding, full_padding)):
        return net

    if set(net.dilation) != {1}:
        raise NotImplementedError(f'Unsupported dilation: {net.dilation}')

    padding = tuple(math.ceil(fp / 2) for fp in full_padding)
    kernel_size = tuple(hp + s + hp for s, hp in zip(net.stride, padding))

    r: nn.modules.conv._ConvNd
    if isinstance(net, nn.modules.conv._ConvTransposeNd):
        r = nn.ConvTranspose2d(
            net.in_channels,
            net.out_channels,
            _tup2(kernel_size),
            _tup2(net.stride),
            _tup2(padding),
            _tup2(net.output_padding),
            net.groups,
            net.bias is not None,
            _tup2(net.dilation),
            net.padding_mode,
        )
    else:
        r = nn.Conv2d(
            net.in_channels,
            net.out_channels,
            _tup2(kernel_size),
            _tup2(net.stride),
            _tup2(padding),
            _tup2(net.dilation),
            net.groups,
            net.bias is not None,
            net.padding_mode,
        )

    with torch.no_grad():
        loc = map(slice, net.kernel_size)
        r.weight.zero_()
        r.weight[(..., *loc)].copy_(net.weight)
        if net.bias is not None and r.bias is not None:
            r.bias.copy_(net.bias)
    return r


# ---------------------------- batch norm fusion -----------------------------


def _fuse_conv_bn_eval[M: nn.Module](
    conv: M,
    bn: nn.Module,
    update_weights: bool = True,
) -> tuple[M, nn.Module]:
    # TODO: allow rebuild without weight update
    if not isinstance(bn, nn.BatchNorm2d):
        return conv, bn

    if update_weights:
        assert isinstance(conv, nn.modules.conv._ConvNd)
        conv = fuse_conv_bn_eval(conv.double(), bn.double()).float()

    ms: list[nn.Module] = []
    # TODO: decouple this
    if _typename(bn) == 'timm.layers.norm_act.BatchNormAct2d':
        if isinstance(bn.drop, nn.Module):
            ms.append(bn.drop)
        if isinstance(bn.act, nn.Module):
            ms.append(bn.act)
    return conv, _build_seq(*ms)


@singledispatch
def fuse_conv_bn(m: nn.Module) -> None:
    """Fuses convolutional layers with subsequent batch normalization layers.

    Reduces computational overhead during inference and can improve
    model performance.

    Works with standard PyTorch modules and extended types from timm & SMP.
    """
    if any(isinstance(c, _BatchNorm) for c in m.children()):
        print(type(m).__module__ + '.' + type(m).__qualname__)
        raise NotImplementedError(type(m))
        # print([type(c).__name__ for c in net.children()
        #        if isinstance(c, _BatchNorm)])


@fuse_conv_bn.register
def _(m: nn.Sequential) -> None:
    i = 0
    conv_tps = tuple(_CONV_TYPES)
    bn_tps = tuple(_BN_TYPES)
    while i < len(m) - 1:
        if isinstance(m[i], conv_tps) and isinstance(m[i + 1], bn_tps):
            m[i], m[i + 1] = _fuse_conv_bn_eval(m[i], m[i + 1])
            if isinstance(m[i + 1], nn.Identity):
                del m[i + 1]
        else:
            i += 1


@fuse_conv_bn.register
def _(m: BasicBlock) -> None:
    m.conv1, m.bn1 = _fuse_conv_bn_eval(m.conv1, m.bn1)
    m.conv2, m.bn2 = _fuse_conv_bn_eval(m.conv2, m.bn2)


@fuse_conv_bn.register
def _(m: Bottleneck) -> None:
    m.conv1, m.bn1 = _fuse_conv_bn_eval(m.conv1, m.bn1)
    m.conv2, m.bn2 = _fuse_conv_bn_eval(m.conv2, m.bn2)
    m.conv3, m.bn3 = _fuse_conv_bn_eval(m.conv3, m.bn3)


# ----------------------------- timm extensions ------------------------------


def _fuse_convbnact(m: 'ConvBnAct') -> None:
    m.conv, m.bn1 = _fuse_conv_bn_eval(m.conv, m.bn1)


def _fuse_dwsepconv(m: 'DepthwiseSeparableConv') -> None:
    m.conv_dw, m.bn1 = _fuse_conv_bn_eval(m.conv_dw, m.bn1)
    m.conv_pw, m.bn2 = _fuse_conv_bn_eval(m.conv_pw, m.bn2)


def _fuse_inverted_residual(m: 'InvertedResidual') -> None:
    m.conv_pw, m.bn1 = _fuse_conv_bn_eval(m.conv_pw, m.bn1)
    m.conv_dw, m.bn2 = _fuse_conv_bn_eval(m.conv_dw, m.bn2)
    m.conv_pwl, m.bn3 = _fuse_conv_bn_eval(m.conv_pwl, m.bn3)


def _fuse_mb3_enet_features(
    m: 'MobileNetV3Features | EfficientNetFeatures',
) -> None:
    m.conv_stem, m.bn1 = _fuse_conv_bn_eval(m.conv_stem, m.bn1)


# ------------------------------ smp extensions ------------------------------


def _fuse_resnet_encoder(m: 'ResNetEncoder') -> None:
    m.conv1, m.bn1 = _fuse_conv_bn_eval(m.conv1, m.bn1)


# ------------------------------ simplification ------------------------------


@singledispatch
def flatten(net: nn.Module) -> nn.Module:
    """
    Recursively flattens nested sequential modules and removes Identity
    layers.

    Preserves non-Sequential modules with their named children.
    """
    children = {name: flatten(m) for name, m in net.named_children()}
    if not _is_true_sequential(net):
        for name, m in children.items():
            setattr(net, name, m)
        return net

    ms = [
        m_
        for m in children.values()
        for m_ in ([*m] if _is_true_sequential(m) else [m])
    ]
    return _build_seq(*ms)


def _unpack_act(m: 'Activation') -> nn.Module:
    return flatten(m.activation)


def _unpack_attn(m: 'Attention') -> nn.Module:
    return flatten(m.attention)


def _unpack_seg_block(m: 'SegmentationBlock') -> nn.Module:
    return flatten(m.block)


def _unpack_conv_gn_gelu(m: 'Conv3x3GNReLU') -> nn.Module:
    block = flatten(m.block)
    if not m.upsample:
        return block

    up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    if isinstance(block, nn.Identity):
        return up
    if _is_true_sequential(block):
        return _build_seq(*block, up)
    raise NotImplementedError(f'Unknown module type: {block}')
    # return nn.Sequential(*cast(nn.Sequential, block), up)


# ------------------------ flatten nested sequentials ------------------------


def _unpack_seq(net: nn.Sequential):
    for m in net.children():
        if _is_true_sequential(m):
            yield from _unpack_seq(m)
        else:
            yield m


def flatten_seq(net: nn.Module) -> nn.Module:
    """
    Flattens nested Sequential modules in-place by unfolding
    all child Sequentials.

    Unlike flatten(), this function modifies the original module structure by
    recursively expanding nested Sequential containers and replacing them with
    a single flattened Sequential. This operation is performed in-place.
    """
    todo = [m for m in net.modules() if _is_true_sequential(m)]
    for m in todo:
        (*gcs,) = _unpack_seq(m)  # got flattened grand-children
        if gcs == [*m]:  # is already flat
            continue
        while len(m):  # make child empty
            del m[-1]
        m.extend(gcs)  # fill child from flat list of grand-children
    return net


# ---------------- remove redundant (w.r.t. inference) module ----------------


def remove_infer_no_ops(net: nn.Module) -> nn.Module:
    """
    Removes modules that have no effect during inference.

    This function recursively traverses the module hierarchy and replaces or
    removes modules that don't affect the forward pass during inference,
    such as:
    - Identity modules
    - Dropout layers (which are only active during training)
    - Empty Sequential containers
    """
    if isinstance(net, _NO_OP_TYPES):
        return nn.Identity()
    ms = {name: remove_infer_no_ops(c) for name, c in net.named_children()}

    if not _is_true_sequential(net):
        for name, m in ms.items():
            setattr(net, name, m)
        return net

    for name, m in ms.items():
        if isinstance(m, nn.Identity):
            delattr(net, name)
        else:
            setattr(net, name, m)
    return net if net else nn.Identity()


# -------------------------------- utilities ---------------------------------


def _build_seq(*ms: nn.Module) -> nn.Module:
    ms = tuple(m for m in ms if not isinstance(m, nn.Identity))
    if not ms:
        return nn.Identity()
    if len(ms) == 1:
        return ms[0]
    return nn.Sequential(*ms)


def _is_true_sequential(m: nn.Module) -> TypeGuard[nn.Sequential]:
    """Checks if module behaves like original nn.Sequential"""
    return isinstance(m, nn.Sequential) and m.__class__.forward is _SEQ_FWD


def _get_effective_kernel(k: int, d: int) -> int:
    return (k - 1) * d + 1


def _get_full_padding(x: int, k: int, s: int) -> int:
    # either non-zero (input % stride),
    # or stride, if input is divisible by stride
    dk = (x % s) or s  # 1..s
    return k - dk
    # return max(0, k - dk)


def _tup2[T](xs: tuple[T, ...]) -> tuple[T, T]:
    assert len(xs) == 2
    return xs[0], xs[1]


def _typename(obj) -> str:
    tp = type(obj)
    return tp.__module__ + '.' + tp.__qualname__


# -------------------------------- hooks -------------------------------------


def _hook_timm_layers(mod) -> None:
    if TYPE_CHECKING:
        import timm.layers as mod  # noqa
    _CONV_TYPES.append(mod.Conv2dSame)
    _BN_TYPES.append(mod.BatchNormAct2d)
    pad_conv_sym_same.register(mod.Conv2dSame, _pad_conv_sym_same)


def _hook_timm_efficientnet(mod) -> None:
    if TYPE_CHECKING:
        import timm.models._efficientnet_blocks as mod  # noqa
    _CONV_TYPES.extend((mod.ConvBnAct, mod.DepthwiseSeparableConv))
    fuse_conv_bn.register(mod.ConvBnAct, _fuse_convbnact)
    fuse_conv_bn.register(mod.InvertedResidual, _fuse_inverted_residual)
    fuse_conv_bn.register(mod.DepthwiseSeparableConv, _fuse_dwsepconv)


def _hook_timm_models(mod) -> None:
    if TYPE_CHECKING:
        import timm.models as mod  # noqa
    fuse_conv_bn.register(mod.MobileNetV3Features, _fuse_mb3_enet_features)
    fuse_conv_bn.register(mod.EfficientNetFeatures, _fuse_mb3_enet_features)


def _hook_smp_base(mod) -> None:
    if TYPE_CHECKING:
        import segmentation_models_pytorch.base.modules as mod  # type: ignore[no-redef]  # noqa
    flatten.register(mod.Activation, _unpack_act)
    flatten.register(mod.Attention, _unpack_attn)


def _hook_smp_resnet(mod) -> None:
    if TYPE_CHECKING:
        import segmentation_models_pytorch.encoders.resnet as mod  # type: ignore[no-redef]  # noqa
    fuse_conv_bn.register(mod.ResNetEncoder, _fuse_resnet_encoder)


def _hook_smp_fpn(mod) -> None:
    if TYPE_CHECKING:
        import segmentation_models_pytorch.decoders.fpn.decoder as mod  # type: ignore[no-redef]  # noqa
    flatten.register(mod.SegmentationBlock, _unpack_seg_block)
    flatten.register(mod.Conv3x3GNReLU, _unpack_conv_gn_gelu)


register_post_import_hook(_hook_timm_layers, 'timm.layers')
register_post_import_hook(
    _hook_timm_efficientnet, 'timm.models._efficientnet_blocks'
)
register_post_import_hook(_hook_timm_models, 'timm.models')
register_post_import_hook(
    _hook_smp_base, 'segmentation_models_pytorch.base.modules'
)
register_post_import_hook(
    _hook_smp_resnet, 'segmentation_models_pytorch.encoders.resnet'
)
register_post_import_hook(
    _hook_smp_fpn, 'segmentation_models_pytorch.decoders.fpn.decoder'
)
