__all__ = [
    'all_reduce',
    'auto_ddp',
    'auto_model',
    'barrier',
    'broadcast_call',
    'get_ddp_info',
]

import pickle
from collections.abc import Callable
from functools import partial, update_wrapper
from multiprocessing.reduction import ForkingPickler
from typing import Any, Concatenate, NamedTuple

import torch
import torch.cuda
import torch.distributed as dist
from torch import Tensor, nn
from torch.multiprocessing.spawn import start_processes

type _TrainFn[**P] = Callable[Concatenate[nn.Module, P], Any]

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


def all_reduce(*tensors: Tensor, mean: bool = False) -> tuple[Tensor, ...]:
    """Reduce tensors across all machines"""
    if (ddp := get_ddp_info()) and ddp.world > 1:
        tensors = tuple(t.clone() for t in tensors)

        ops = [dist.all_reduce(t, async_op=True) for t in tensors]
        for op in ops:
            op.wait()

        if mean:
            tensors = tuple(t / ddp.world for t in tensors)
    return tensors


# --------------------------------- wrappers ---------------------------------


def auto_model(net: nn.Module, sync_bn: bool = True) -> nn.Module:
    if (ddp := get_ddp_info()) and ddp.world > 1:
        torch.cuda.set_device(ddp.rank)

        net.to(ddp.rank)
        if sync_bn:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        return nn.parallel.DistributedDataParallel(net, device_ids=[ddp.rank])

    net.cuda()
    return (
        nn.parallel.DataParallel(net) if torch.cuda.device_count() > 1 else net
    )


class _AutoDdp[**P]:
    def __init__(
        self,
        train_fn: _TrainFn[P],
        net: nn.Module,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
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
        start_processes(self._worker, nprocs=self.ngpus)

    def _worker(self, rank: int | None) -> None:
        if rank is None:
            return self.train_fn(self.net, *self.args, **self.kwargs)

        dist.init_process_group('nccl', world_size=self.ngpus, rank=rank)
        try:
            self.train_fn(auto_model(self.net), *self.args, **self.kwargs)
        finally:
            dist.destroy_process_group()


def auto_ddp[**P](train_fn: _TrainFn[P]) -> _TrainFn[P]:
    return update_wrapper(partial(_AutoDdp, train_fn), train_fn)


def broadcast_call[**P, R](fn: Callable[P, R], /) -> Callable[P, R]:
    """
    Callable will be called in single process,
    and its result will be broadcasted to all the neighbours.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
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

            assert handles[
                0
            ], '"torch.distributed.broadcast_object_list" failed'
            result = pickle.loads(handles[0])

        return result

    return update_wrapper(wrapper, fn)
