__all__ = [
    'flatten', 'flatten_seq', 'fuse_conv_bn', 'pad_conv_sym_same',
    'remove_infer_no_ops'
]
import math
from functools import partial, singledispatch
from typing import TYPE_CHECKING, TypeGuard, cast

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models.resnet import BasicBlock, Bottleneck
from wrapt import register_post_import_hook

if TYPE_CHECKING:
    from segmentation_models_pytorch.base.modules import Activation, Attention
    from segmentation_models_pytorch.decoders.fpn.decoder import (
        Conv3x3GNReLU, SegmentationBlock)
    from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
    from timm.models import EfficientNetFeatures, MobileNetV3Features
    from timm.models.efficientnet_blocks import (
        ConvBnAct, DepthwiseSeparableConv, InvertedResidual)

_SEQ_FWD = nn.Sequential.forward
_CONV_TYPES: list[type] = [nn.Conv2d]
_BN_TYPES: list[type] = [nn.BatchNorm2d]
_NO_OP_TYPES: list[type] = [nn.Identity, nn.modules.dropout._DropoutNd]

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
    eff_kernel = *map(_get_effective_kernel, net.kernel_size, net.dilation),
    full_padding = *map(partial(_get_full_padding, 0), eff_kernel, net.stride),
    # Check if padding is symmetric
    if all(hp * 2 == fp for hp, fp in zip(net.padding, full_padding)):
        return net

    if set(net.dilation) != {1}:
        raise NotImplementedError(f'Unsupported dilation: {net.dilation}')

    padding = *(math.ceil(fp / 2) for fp in full_padding),
    kernel_size = *(hp + s + hp for s, hp in zip(net.stride, padding)),

    r: nn.modules.conv._ConvNd
    if isinstance(net, nn.modules.conv._ConvTransposeNd):
        r = nn.ConvTranspose2d(
            net.in_channels,
            net.out_channels,
            kernel_size,  # type: ignore[arg-type]
            net.stride,  # type: ignore[arg-type]
            padding,  # type: ignore[arg-type]
            net.output_padding,  # type: ignore[arg-type]
            net.groups,
            net.bias is not None,
            net.dilation,  # type: ignore[arg-type]
            net.padding_mode,
        )
    else:
        r = nn.Conv2d(
            net.in_channels,
            net.out_channels,
            kernel_size,  # type: ignore[arg-type]
            net.stride,  # type: ignore[arg-type]
            padding,  # type: ignore[arg-type]
            net.dilation,  # type: ignore[arg-type]
            net.groups,
            net.bias is not None,
            net.padding_mode,
        )

    with torch.no_grad():
        loc = ..., *map(slice, net.kernel_size)
        r.weight.zero_()
        r.weight[loc].copy_(net.weight)
        if net.bias is not None and r.bias is not None:
            r.bias.copy_(net.bias)
    return r


# ---------------------------- batch norm fusion -----------------------------


def _fuse_conv_bn_eval(conv, bn, update_weights=True):
    # TODO: allow rebuild without weight update
    if not isinstance(bn, nn.BatchNorm2d):
        return conv, bn

    if update_weights:
        conv = fuse_conv_bn_eval(conv.double(), bn.double()).float()

    ms: list[nn.Module] = []
    # TODO: decouple this
    if _typename(bn) == 'timm.models.layers.norm_act.BatchNormAct2d':
        if not isinstance(bn.drop, nn.Identity):
            ms.append(bn.drop)  # type: ignore[arg-type]
        if not isinstance(bn.act, nn.Identity):
            ms.append(bn.act)  # type: ignore[arg-type]
    if not ms:
        return conv, nn.Identity()
    if len(ms) == 1:
        return conv, ms[0]
    return conv, nn.Sequential(*ms)


@singledispatch
def fuse_conv_bn(m: nn.Module) -> None:
    if any(isinstance(c, _BatchNorm) for c in m.children()):
        print(type(m).__module__ + '.' + type(m).__qualname__)
        raise NotImplementedError(type(m))
        # print([type(c).__name__ for c in net.children()
        #        if isinstance(c, _BatchNorm)])


@fuse_conv_bn.register
def _(m: nn.Sequential):
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
def _(m: BasicBlock):
    m.conv1, m.bn1 = _fuse_conv_bn_eval(m.conv1, m.bn1)
    m.conv2, m.bn2 = _fuse_conv_bn_eval(m.conv2, m.bn2)


@fuse_conv_bn.register
def _(m: Bottleneck):
    m.conv1, m.bn1 = _fuse_conv_bn_eval(m.conv1, m.bn1)
    m.conv2, m.bn2 = _fuse_conv_bn_eval(m.conv2, m.bn2)
    m.conv3, m.bn3 = _fuse_conv_bn_eval(m.conv3, m.bn3)


# ----------------------------- timm extensions ------------------------------


def _fuse_convbnact(m: 'ConvBnAct'):
    m.conv, m.bn1 = _fuse_conv_bn_eval(m.conv, m.bn1)


def _fuse_dwsepconv(m: 'DepthwiseSeparableConv'):
    m.conv_dw, m.bn1 = _fuse_conv_bn_eval(m.conv_dw, m.bn1)
    m.conv_pw, m.bn2 = _fuse_conv_bn_eval(m.conv_pw, m.bn2)


def _fuse_inverted_residual(m: 'InvertedResidual'):
    m.conv_pw, m.bn1 = _fuse_conv_bn_eval(m.conv_pw, m.bn1)
    m.conv_dw, m.bn2 = _fuse_conv_bn_eval(m.conv_dw, m.bn2)
    m.conv_pwl, m.bn3 = _fuse_conv_bn_eval(m.conv_pwl, m.bn3)


def _fuse_mb3_enet_features(m: 'MobileNetV3Features | EfficientNetFeatures'):
    m.conv_stem, m.bn1 = _fuse_conv_bn_eval(m.conv_stem, m.bn1)


# ------------------------------ smp extensions ------------------------------


def _fuse_resnet_encoder(m: 'ResNetEncoder'):
    m.conv1, m.bn1 = _fuse_conv_bn_eval(m.conv1, m.bn1)


# ------------------------------ simplification ------------------------------


@singledispatch
def flatten(net: nn.Module) -> nn.Module:
    children = {name: flatten(m) for name, m in net.named_children()}
    if not _is_true_sequential(net):
        for name, m in children.items():
            setattr(net, name, m)
        return net

    ms = [
        m_ for m in children.values()
        for m_ in ([*m] if _is_true_sequential(m) else [m])
        if not isinstance(m_, nn.Identity)
    ]
    if not ms:
        return nn.Identity()
    if len(ms) == 1:
        return ms[0]
    return nn.Sequential(*ms)


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
    return nn.Sequential(
        *cast(nn.Sequential, block),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    )


# ------------------------ flatten nested sequentials ------------------------


def _unpack_seq(net: nn.Sequential):
    for m in net.children():
        if _is_true_sequential(m):
            yield from _unpack_seq(m)
        else:
            yield m


def flatten_seq(net: nn.Module) -> nn.Module:
    todo = [m for m in net.modules() if _is_true_sequential(m)]
    for m in todo:
        *gcs, = _unpack_seq(m)  # got flattened grand-children
        if gcs == [*m]:  # is already flat
            continue
        while len(m):  # make child empty
            del m[-1]
        m.extend(gcs)  # fill child from flat list of grand-children
    return net


# ---------------- remove redundant (w.r.t. inference) module ----------------


def remove_infer_no_ops(net: nn.Module) -> nn.Module:
    if isinstance(net, tuple(_NO_OP_TYPES)):
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


def _is_true_sequential(m: nn.Module) -> TypeGuard[nn.Sequential]:
    """Checks if module behaves like original nn.Sequential"""
    return isinstance(m, nn.Sequential) and type(m).forward is _SEQ_FWD


def _get_effective_kernel(k: int, d: int) -> int:
    return (k - 1) * d + 1


def _get_full_padding(x: int, k: int, s: int) -> int:
    # either non-zero (input % stride),
    # or stride, if input is divisible by stride
    dk = (x % s) or s  # 1..s
    return k - dk
    # return max(0, k - dk)


def _typename(obj) -> str:
    tp = type(obj)
    return tp.__module__ + '.' + tp.__qualname__


# -------------------------------- hooks -------------------------------------


def _hook_timm(timm):
    if TYPE_CHECKING:
        import timm  # type: ignore[no-redef]
        import timm.models.efficientnet_blocks  # type: ignore[no-redef]

    models = timm.models
    layers = models.layers
    enb = models.efficientnet_blocks

    _CONV_TYPES.extend(
        (layers.Conv2dSame, enb.ConvBnAct, enb.DepthwiseSeparableConv))
    _BN_TYPES.append(layers.BatchNormAct2d)

    fuse_conv_bn.register(enb.ConvBnAct, _fuse_convbnact)
    fuse_conv_bn.register(enb.InvertedResidual, _fuse_inverted_residual)
    fuse_conv_bn.register(enb.DepthwiseSeparableConv, _fuse_dwsepconv)

    fuse_conv_bn.register(models.MobileNetV3Features, _fuse_mb3_enet_features)
    fuse_conv_bn.register(models.EfficientNetFeatures, _fuse_mb3_enet_features)

    pad_conv_sym_same.register(layers.Conv2dSame, _pad_conv_sym_same)


def _hook_smp(smp):
    if TYPE_CHECKING:
        import segmentation_models_pytorch as smp  # type: ignore[no-redef]

        # import segmentation_models_pytorch.base.modules

    flatten.register(smp.base.modules.Activation, _unpack_act)
    flatten.register(smp.base.modules.Attention, _unpack_attn)
    flatten.register(smp.decoders.fpn.decoder.SegmentationBlock,
                     _unpack_seg_block)
    flatten.register(smp.decoders.fpn.decoder.Conv3x3GNReLU,
                     _unpack_conv_gn_gelu)
    fuse_conv_bn.register(smp.encoders.resnet.ResNetEncoder,
                          _fuse_resnet_encoder)


register_post_import_hook(_hook_timm, 'timm')
register_post_import_hook(_hook_smp, 'segmentation_models_pytorch')
