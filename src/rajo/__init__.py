__all__ = [
    'Trainer', 'data', 'detach_', 'device', 'dump_to_onnx', 'eval_', 'frozen',
    'get_gpu_capability', 'get_gpu_memory_info', 'get_grads', 'get_loader',
    'inference', 'materialize', 'metrics', 'nn', 'optim', 'param_count',
    'plot_model', 'profile'
]

from importlib import import_module
from typing import TYPE_CHECKING

from . import data, metrics, nn, optim
from ._lazy import materialize
from ._trainer import Trainer
from .amp import get_grads
from .data._loader import get_loader
from .util import (detach_, device, dump_to_onnx, eval_, frozen, inference,
                   param_count, profile)

_exports = {
    '.plot': ['plot_model'],
    '.driver': ['get_gpu_capability', 'get_gpu_memory_info'],
}
_submodule_by_name = {
    name: modname for modname, names in _exports.items() for name in names
}

if TYPE_CHECKING:
    from .driver import get_gpu_capability, get_gpu_memory_info
    from .plot import plot_model

else:

    def __getattr__(name: str):
        if modname := _submodule_by_name.get(name):
            mod = import_module(modname, __package__)
            globals()[name] = obj = getattr(mod, name)
            return obj
        raise AttributeError(f'No attribute {name}')

    def __dir__():
        return __all__
