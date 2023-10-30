__all__ = [
    'auto_ddp', 'auto_model', 'barrier', 'get_ddp_info', 'reduce_if_needed'
]

import pickle
from collections.abc import Callable
from functools import partial, update_wrapper
from typing import Any, NamedTuple, Protocol, TypeVar, cast, overload

import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as tmp
from torch import Tensor, nn
from torch.multiprocessing.reductions import ForkingPickler

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
def reduce_if_needed(*values: int | float,
                     mean: bool = ...) -> tuple[float, ...]:
    ...


@overload
def reduce_if_needed(*values: Tensor, mean: bool = ...) -> tuple[Tensor, ...]:
    ...


def reduce_if_needed(*values: Tensor | float | int,
                     mean: bool = False) -> tuple[Tensor | float, ...]:
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


class _TrainFn(Protocol):
    def __call__(self, __net: nn.Module, *args, **kwargs) -> Any:
        ...


_F = TypeVar('_F', bound=Callable)
_TrainFnType = TypeVar('_TrainFnType', bound=_TrainFn)


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

        dist.init_process_group(
            backend='nccl', rank=rank, world_size=self.ngpus)
        try:
            self.train_fn(auto_model(self.net), *self.args, **self.kwargs)
        finally:
            dist.destroy_process_group()


def auto_ddp(train_fn: _TrainFnType) -> _TrainFnType:
    return cast(_TrainFnType,
                update_wrapper(partial(_AutoDdp, train_fn), train_fn))


def once_per_world(fn: _F) -> _F:
    """Call function only in rank=0 process, and share result for others"""
    def wrapper(*args, **kwargs):
        ddp = get_ddp_info()
        if not ddp or ddp.world == 1:
            # Master process, so no neighbors to share results with
            return fn(*args, **kwargs)

        if ddp.rank == 0:  # 0th child
            result = fn(*args, **kwargs)
            handles = [bytes(ForkingPickler.dumps(result))]
        else:
            result = None
            handles = [None]

        # Send all from 0 to rest
        dist.broadcast_object_list(handles, src=0)

        if ddp.rank != 0:  # Rebuild from serialized
            assert handles[0] is not None
            result = pickle.loads(handles[0])

        return result

    return cast(_F, update_wrapper(wrapper, fn))
