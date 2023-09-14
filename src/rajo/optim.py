__all__ = ['AdamW', 'Lion', 'RAdam', 'SGDW']

from collections.abc import Callable, Iterable, Sequence
from dataclasses import InitVar, asdict, dataclass, field
from typing import NamedTuple, final

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer

from . import _foreach

# TODO: support Scaler's is_inf


class _GradState(NamedTuple):
    p: Parameter
    grad: Tensor
    state: list[Tensor]


@dataclass
class SingleGroupOptimizer(Optimizer):
    params: InitVar[list[Parameter]]
    states: dict[Parameter, list[Tensor]] = field(init=False)
    _step = 0

    def __post_init__(self, params: list[Parameter]):
        self.states = {p: [] for p in params}

    @final
    def zero_grad(self, set_to_none: bool = True) -> None:
        if set_to_none:
            for p in self.states:
                p.grad = None
        else:
            for states in self._state_groups():
                _foreach.zero_(s.grad for s in states)

    def common_step_args(self) -> tuple:
        return ()

    def __getstate__(self):
        raise NotImplementedError

    @final
    def state_dict(self):
        return asdict(self) | {
            '_step': self._step,
            'grads': [p.grad for p in self.states],
            'states': [*self.states.values()],
        }

    @final
    @torch.no_grad()
    def load_state_dict(self, state: dict):
        self.__dict__.update({
            k: v for k, v in state.items() if k not in ('grads', 'states')
        })
        for p, g in zip(self.states, state['grads']):
            if g is None:
                p.grad = None
            elif p.grad is None:
                p.grad = g.clone().detach_().to(p.device, non_blocking=True)
            else:
                p.grad.copy_(g.to(p.device, non_blocking=True, copy=False))

        for dst, src in zip(self.states.values(), state['states']):
            dst[:] = src

    def _state_groups(self) -> Iterable[list[_GradState]]:
        r: dict[tuple, list[_GradState]] = {}
        for p, ts in self.states.items():
            if p.grad is None:
                continue
            num_values = sum(t is not None for t in ts)
            r.setdefault((p.device, p.dtype, p.grad.is_sparse, num_values),
                         []).append(_GradState(p, p.grad, ts))

        return r.values()

    @final
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self._step += 1
        loss = closure() if closure else None

        args = self.common_step_args()
        for state_groups in self._state_groups():
            with torch.no_grad():
                self.device_step(state_groups, *args)

        return loss

    def device_step(self, states: Sequence[_GradState], *args):
        raise NotImplementedError


@dataclass
class SGDW(SingleGroupOptimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    """
    lr: float = 0.003
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False

    def __post_init__(self, params):
        assert self.lr >= 0
        assert self.momentum >= 0
        assert self.weight_decay >= 0
        assert (not self.nesterov
                or (self.momentum > 0 and self.dampening == 0)
                ), 'Nesterov momentum requires a momentum and zero dampening'
        super().__post_init__(params)

    def device_step(self, states: Sequence[_GradState], *args):
        params = [s.p for s in states]
        grads = [s.grad for s in states]

        if self.weight_decay != 0:
            _foreach.mul_(params, scalar=1 - self.lr * self.weight_decay)

        if self.momentum != 0:
            if states[0].state:
                bufs = [s.state[0] for s in states]
                _foreach.mul_(bufs, scalar=self.momentum)
                _foreach.add_(bufs, grads, alpha=1 - self.dampening)
            else:
                for s in states:
                    s.state.append(s.grad.clone().detach_())
                bufs = [s.state[0] for s in states]

            if self.nesterov:
                _foreach.add_(grads, bufs, alpha=self.momentum)
            else:
                grads = bufs

        _foreach.add_(params, grads, alpha=-self.lr)


@dataclass
class AdamW(SingleGroupOptimizer):
    r"""Implements AdamW algorithm.

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False

    def __post_init__(self, params):
        assert self.lr >= 0.0
        assert self.eps >= 0.0
        for i, beta in enumerate(self.betas):
            assert 0.0 <= beta < 1.0, \
                f'Invalid beta at index {i}: {self.betas}'
        super().__post_init__(params)

    def common_step_args(self) -> tuple:
        beta1, beta2 = self.betas
        bias_correction1 = 1 - beta1 ** self._step
        bias_correction2 = 1 - beta2 ** self._step
        return self.lr * (bias_correction2 ** 0.5) / bias_correction1,

    def device_step(self, states: Sequence[_GradState], step_size=1, *args):
        params = [s.p for s in states]
        grads = [s.grad for s in states]

        num_bufs = 3 if self.amsgrad else 2
        for p, s in zip(params, states):
            if not s.state:
                s.state.extend(torch.zeros_like(p) for _ in range(num_bufs))

        avg, avg_sq, *opt_max_avg_sq = zip(*(s.state for s in states))
        beta1, beta2 = self.betas

        _foreach.lerp_(avg, grads, weight=1 - beta1)

        _foreach.mul_(avg_sq, scalar=beta2)
        _foreach.addcmul_(avg_sq, grads, grads, value=1 - beta2)

        if self.weight_decay != 0:
            _foreach.mul_(params, scalar=1 - self.lr * self.weight_decay)

        if self.amsgrad:
            max_avg_sq = opt_max_avg_sq[0]
            _foreach.maximum_(max_avg_sq, avg_sq)
            avg_sq = max_avg_sq

        denom = _foreach.sqrt(avg_sq)
        _foreach.add_(denom, self.eps)
        _foreach.addcdiv_(params, avg, denom, value=-step_size)


@dataclass
class RAdam(SingleGroupOptimizer):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0
    decay_to_sgd: bool = True
    eps: float = 1e-8

    def common_step_args(self) -> tuple:
        beta1, beta2 = self.betas
        n_sma_max = 2 / (1 - beta2) - 1

        bias_correction1 = 1 - beta1 ** self._step
        bias_correction2 = 1 - beta2 ** self._step

        beta2_t = beta2 ** self._step
        n_sma = n_sma_max - 2 * self._step * beta2_t / bias_correction2

        # more conservative since it's an approximated value
        # variance is not tractable
        if n_sma < 5:
            return False, (1 / bias_correction1)

        k = (n_sma - 4) * (n_sma - 2) / n_sma
        k_max = (n_sma_max - 4) * (n_sma_max - 2) / n_sma_max
        step_size = ((1 - beta2_t) * k / k_max) ** 0.5 / bias_correction1
        return True, step_size

    def device_step(self,
                    states: Sequence[_GradState],
                    is_tractable=False,
                    step_size=1,
                    *args):
        params = [s.p for s in states]
        grads = [s.grad for s in states]
        for p, s in zip(params, states):
            if not s.state:
                s.state.extend(torch.zeros_like(p) for _ in range(2))

        avg, avg_sq = zip(*(s.state for s in states))
        beta1, beta2 = self.betas

        _foreach.lerp_(avg, grads, weight=1 - beta1)

        _foreach.mul_(avg_sq, scalar=beta2)
        _foreach.addcmul_(avg_sq, grads, grads, value=1 - beta2)

        if self.weight_decay != 0:
            _foreach.mul_(params, scalar=1 - self.weight_decay * self.lr)

        if is_tractable:
            denom = _foreach.sqrt(avg_sq)
            _foreach.add_(denom, self.eps)
            _foreach.addcdiv_(params, avg, denom, value=-step_size * self.lr)

        elif self.decay_to_sgd:
            _foreach.add_(params, avg, alpha=-step_size * self.lr)


@dataclass
class Lion(SingleGroupOptimizer):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0

    def device_step(self, states: Sequence[_GradState], *args):
        param = [s.p for s in states]
        grad = [s.grad for s in states]
        for p, s in zip(param, states):
            if not s.state:
                s.state.append(torch.zeros_like(p))

        avg = [s.state[0] for s in states]
        beta1, beta2 = self.betas

        if self.weight_decay != 0:
            _foreach.mul_(param, scalar=1 - self.lr * self.weight_decay)

        update = _foreach.lerp(avg, grad, weight=1 - beta1)
        for u in update:
            u.sign_()
        _foreach.add_(param, update, alpha=-self.lr)

        _foreach.lerp_(avg, grad, weight=1 - beta2)
