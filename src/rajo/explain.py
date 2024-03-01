__all__ = ['module_distribution', 'receptive_field_test']

from copy import deepcopy

import torch
from torch import nn


def module_distribution(net: nn.Module) -> dict[type[nn.Module], int]:
    s: dict[type[nn.Module], int] = {}
    for m in net.modules():
        k = type(m)
        s[k] = s.get(k, 0) + 1
    return s


def receptive_field_test(net: nn.Module,
                         num_channels: int,
                         size: int,
                         device: torch.device | str | None = None):
    net = deepcopy(net)
    set_to = {
        'weight': 1,
        'bias': 0,
        'running_var': 1,
        'running_mean': 0,
    }
    for m in (net.named_parameters, net.named_buffers):
        for name, p in m():
            if not p.is_floating_point():
                continue
            stem = name.split('.')[-1]
            with torch.no_grad():
                p.fill_(set_to[stem])

    shape = (1, num_channels, size, size)
    x = torch.zeros(shape, device=device)
    x[0, :, size // 2, size // 2] = 1

    with torch.no_grad():
        return [net.train(mode)(x) for mode in (False, True)]
