__all__ = ['plot_model']

import functools
from collections.abc import Iterable, Iterator, Mapping, Callable
from contextlib import ExitStack
from typing import Any

import graphviz
import torch
from glow import countable, mangle, si
from torch import Tensor, nn
from torch.autograd.graph import Node

# TODO: Still buggy, continue research/refactor


def flatten(xs) -> Iterator[Tensor]:
    if xs is None:
        return
    if isinstance(xs, Tensor):
        yield xs
        return
    if isinstance(xs, Mapping):
        xs = xs.items()
    if isinstance(xs, Iterable):
        for x in xs:
            yield from flatten(x)
        return
    raise TypeError(f'Unsupported argument type: {type(xs)}')


def sized(t: Tensor) -> str:
    shape = f'{tuple(t.shape)}'
    return shape if t.squeeze().ndim <= 1 else f'{shape}\n{si(t.numel())}'


class Builder:
    def __init__(
        self,
        inputs: list[Tensor],
        params: dict[str, Any],
        *,
        nesting: bool = True,
        variables: bool = True,
    ) -> None:
        # For self._id
        self._buf_count = _tensor_countable()
        self._var_count = _tensor_countable()
        self._obj_count = countable()

        self.inputs = {self._id(var) for var in inputs}
        self.params = {self._id(var): name for name, var in params.items()}

        self.nesting = nesting
        self.variables = variables

        self._mangle = mangle()
        self._memo: dict[str, str] = {}
        self._shapes: dict[Node, tuple[int, ...]] = {}
        root = graphviz.Digraph(
            name='root',
            graph_attr={
                'rankdir': 'LR',
                'newrank': 'true',
                'color': 'lightgrey',
            },
            edge_attr={
                'labelfloat': 'true',
            },
            node_attr={
                'shape': 'box',
                'style': 'filled',
                'fillcolor': 'lightgrey',
                'fontsize': '12',
                'height': '0.2',
                'ranksep': '0.1',
            },
        )
        self.stack = [root]

    def _add_op_node(self, grad_id: str, grad: Node) -> None:
        label = type(grad).__name__.replace('Backward', '')
        if grad in self._shapes:
            label = f'{label}\n-> {tuple(self._shapes[grad])}'
        self.stack[-1].node(grad_id, label)

    def _add_var_node(self, var_id: str, var: Tensor) -> None:
        label = f'{var_id}\n{sized(var)}'

        if param_name := self.params.get(var_id):
            sg = self.stack[-1]
            parts = param_name.split('.')
            if self.nesting:
                label = f'{parts[-1]}\n{label}'
            else:
                label = f'{'.'.join(parts[1:])}\n{label}'
        else:
            sg = self.stack[0]  # unnamed, thus use top level graph

        color = 'yellow' if var_id in self.inputs else 'lightblue'
        sg.node(var_id, label, fillcolor=color)

    def _traverse_saved(self, grad_id: str, *tensors: Tensor) -> None:
        tensors = tuple(v for v in tensors if isinstance(v, Tensor))
        if not tensors:
            return
        sg = graphviz.Digraph()
        sg.attr(rank='same')
        for var in tensors:
            var_id = self._id(var)
            if var_id not in self._memo:
                label = f'{var_id}\n{sized(var)}'
                sg.node(var_id, label, fillcolor='orange')
            sg.edge(var_id, grad_id)
        self.stack[-1].subgraph(sg)

    def _traverse(
        self,
        grad: Node | None,
        depth: int = 0,
    ) -> Iterator[tuple[int, Node, Node]]:
        if grad is None or (grad_id := self._id(grad)) in self._memo:
            return

        g = self.stack[-1]
        assert g.name is not None
        self._memo[grad_id] = head = g.name

        if hasattr(grad, 'variable'):
            # Has variable, so it's either Parameter or Variable
            self._add_var_node(grad_id, grad.variable)
            # yield (depth - 1, None, grad)  # <- only to pass `depth` info
            return

        # Doesn't have variable, so it's "operation"
        self._add_op_node(grad_id, grad)

        # TODO : add merging of tensors with same data
        if self.variables and hasattr(grad, 'saved_tensors'):
            self._traverse_saved(grad_id, *(grad.saved_tensors or ()))

        for grad_next, _ in grad.next_functions:
            if grad_next is None:
                continue
            yield from self._traverse(grad_next, depth + 1)

            next_id = self._id(grad_next)
            tail = self._memo.get(next_id)
            if (
                tail is not None
                and head is not None
                and not (head.startswith(tail) or tail.startswith(head))
            ):
                yield (depth, grad_next, grad)  # leafs, yield for depth-check
                continue

            name = self.params.get(next_id)
            if self.nesting and name and name.rpartition('.')[0] == head:
                sg = graphviz.Digraph()
                sg.attr(rank='same')
                sg.edge(next_id, grad_id)  # same module, same rank
                g.subgraph(sg)
            else:
                self.stack[0].edge(next_id, grad_id)

    def _mark(self, ts) -> None:
        edges: list[tuple[int, Node, Node]] = []
        for t in flatten(ts):
            if t.grad_fn is not None:
                self._shapes[t.grad_fn] = t.shape
                edges += self._traverse(t.grad_fn)
        if not edges:
            return

        max_depth = max(depth for depth, *_ in edges) + 1
        for depth, tail, head in edges:  # inter-module edges
            minlen = f'{max_depth - depth}' if self.nesting else None
            self.stack[0].edge(self._id(tail), self._id(head), minlen=minlen)

    def enter(self, name: str, module: nn.Module, xs) -> None:
        self._mark(xs)
        # -------- start node --------
        if not self.nesting:
            return
        scope = graphviz.Digraph(name=self._mangle(name))
        scope.attr(label=f'{name.split(".")[-1]}:{type(module).__name__}')
        self.stack.append(scope)

    def exit_(self, module: nn.Module, _, ys) -> None:
        self._mark(ys)
        if not self.nesting:
            return
        # Link child with parent and transform to cluster only if child is done
        scope = self.stack.pop()
        scope.name = f'cluster_{scope.name}'
        self.stack[-1].subgraph(scope)
        # -------- end node --------

    def _id(self, x) -> str:
        if hasattr(x, 'variable'):
            x = x.variable
        return (
            (
                f'#{self._var_count(x)}'
                if x.requires_grad
                else f'##{self._buf_count(x)}'
            )
            if isinstance(x, Tensor)
            else f'op{self._obj_count(x)}'
        )


def plot_model(
    model: nn.Module,
    *input_shapes: tuple[int, ...],
    device='cpu',
    nesting: bool = True,
    variables: bool = False,
) -> graphviz.Digraph:
    """Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    """
    inputs = [
        torch.zeros(1, *s, device=device, requires_grad=True)
        for s in input_shapes
    ]
    params = model.state_dict(prefix='root.', keep_vars=True)
    hk = Builder(inputs, params, nesting=nesting, variables=variables)
    with ExitStack() as stack:
        for name, m in model.named_modules(prefix='root'):
            enter_ = functools.partial(hk.enter, name)
            stack.callback(m.register_forward_pre_hook(enter_).remove)
            stack.callback(m.register_forward_hook(hk.exit_).remove)
        model(*inputs)

    dot = hk.stack.pop()
    assert not hk.stack

    dot.filename = getattr(model, 'name', type(model).__qualname__)
    dot.directory = 'graphs'
    dot.format = 'svg'

    size_min = 12
    scale_factor = 0.15
    size = max(size_min, len(dot.body) * scale_factor)

    dot.graph_attr.update(size=f'{size},{size}')
    dot.render(cleanup=True)
    return dot


def _tensor_countable() -> Callable[[Tensor], int]:
    """`countable` with `torch.data_ptr` instead of `id`"""
    instances: dict[int, int] = {}
    return lambda obj: instances.setdefault(obj.data_ptr(), len(instances))
