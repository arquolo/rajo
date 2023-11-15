__all__ = [
    'all_reduce', 'auto_ddp', 'auto_model', 'barrier', 'broadcast_call',
    'get_ddp_info', 'once_per_world', 'reduce_if_needed'
]

import pickle
import warnings
from collections.abc import Callable
from functools import partial, update_wrapper
from typing import (TYPE_CHECKING, Any, Concatenate, Literal, NamedTuple,
                    ParamSpec, TypeVar, overload)

import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as tmp
from torch import Tensor, nn
from torch.multiprocessing.reductions import ForkingPickler

_P = ParamSpec('_P')
_R = TypeVar('_R')
_Number = TypeVar('_Number', int, float)
_TrainFn = Callable[Concatenate[nn.Module, _P], Any]

# -------------------------------- primitives --------------------------------


class _DdpInfo(NamedTuple):
    world: int
    rank: int


def get_ddp_info() -> _DdpInfo | None:
    if not dist.is_initialized():
        return None
    return _DdpInfo(dist.get_world_size(), dist.get_rank())


def barrier(rank: int | None = None) -> None:
    """Synchronize all processes"""
    if (info := get_ddp_info()) and (rank is None or rank == info.rank):
        dist.barrier()


@overload
def all_reduce(*values: _Number,
               mean: Literal[False] = ...) -> tuple[_Number, ...]:
    ...


@overload
def all_reduce(*values: int | float, mean: bool) -> tuple[float, ...]:
    ...


@overload
def all_reduce(*values: Tensor, mean: bool = ...) -> tuple[Tensor, ...]:
    ...


def all_reduce(*values: Tensor | float | int,
               mean: bool = False) -> tuple[Tensor | float | int, ...]:
    """Reduce tensors across all machines"""
    if (ddp := get_ddp_info()) and ddp.world > 1:
        device_id = torch.cuda.current_device()
        device = torch.device(f'cuda:{device_id}')

        pairs = [((v.clone(), False) if isinstance(v, Tensor) else
                  (torch.tensor(v, device=device), True)) for v in values]
        tensors: tuple[Tensor, ...]
        unpack: tuple[bool, ...]
        tensors, unpack = zip(*pairs)

        ops = [dist.all_reduce(t, async_op=True) for t in tensors]
        for op in ops:
            op.wait()  # type: ignore

        if mean:
            tensors = *(t / ddp.world for t in tensors),
        values = *(t.item() if u else t for t, u in zip(tensors, unpack)),
    return values


# --------------------------------- wrappers ---------------------------------


def auto_model(net: nn.Module, sync_bn: bool = True) -> nn.Module:
    if (ddp := get_ddp_info()) and ddp.world > 1:
        torch.cuda.set_device(ddp.rank)

        net.to(ddp.rank)
        if sync_bn:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        return nn.parallel.DistributedDataParallel(net, device_ids=[ddp.rank])

    net.cuda()
    return (nn.parallel.DataParallel(net)
            if torch.cuda.device_count() > 1 else net)


class _AutoDdp:
    def __init__(self, train_fn: _TrainFn, net: nn.Module, *args, **kwargs):
        self.train_fn = train_fn
        self.net = net
        self.args = args
        self.kwargs = kwargs
        self.ngpus = torch.cuda.device_count()

        if self.ngpus == 1:
            self._worker(None)
            return

        # ! Not tested
        # * Actually, here we can use loky.ProcessPoolExecutor, like this:
        # from glow import map_n
        # ngpus = self.ngpus
        # jobs = map_n(self._worker, range(ngpus), max_workers=ngpus, mp=True)
        # list(jobs)
        # * Left as safe measure
        tmp.spawn(self._worker, nprocs=self.ngpus)

    def _worker(self, rank: int | None) -> None:
        if rank is None:
            return self.train_fn(self.net, *self.args, **self.kwargs)

        dist.init_process_group('nccl', world_size=self.ngpus, rank=rank)
        try:
            self.train_fn(auto_model(self.net), *self.args, **self.kwargs)
        finally:
            dist.destroy_process_group()


def auto_ddp(
    train_fn: Callable[Concatenate[nn.Module, _P], Any]
) -> Callable[Concatenate[nn.Module, _P], Any]:
    return update_wrapper(partial(_AutoDdp, train_fn), train_fn)


def broadcast_call(fn: Callable[_P, _R], /) -> Callable[_P, _R]:
    """
    Callable will be called in single process,
    and its result will be broadcasted to all the neighbours.
    """
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        ddp = get_ddp_info()
        if not ddp or ddp.world == 1:
            # Master process, so no neighbors to share results with
            return fn(*args, **kwargs)

        if ddp.rank == 0:  # Call and broadcast result to all neighbours
            result = fn(*args, **kwargs)
            handles = [bytes(ForkingPickler.dumps(result))]
            dist.broadcast_object_list(handles, src=0)

        else:  # Gather result from #0
            handles = [b'']
            dist.broadcast_object_list(handles, src=0)

            assert handles[0], \
                '"torch.distributed.broadcast_object_list" failed'
            result = pickle.loads(handles[0])

        return result

    return update_wrapper(wrapper, fn)


# -------------------------------- 0.2.x API ---------------------------------
# TODO: remove in 0.3.x release

if TYPE_CHECKING:
    reduce_if_needed = all_reduce
    once_per_world = broadcast_call

_deprecations = {
    'reduce_if_needed': ('all_reduce', all_reduce),
    'once_per_world': ('broadcast_call', broadcast_call),
}


def __getattr__(name: str):
    if new := _deprecations.get(name):
        new_name, new_attr = new
        warnings.warn(
            f'"rajo.distributed.{name}" is deprecated. '
            f'Use "rajo.distributed.{new_name}" instead',
            category=DeprecationWarning,
            stacklevel=2)
        return new_attr
    raise AttributeError
