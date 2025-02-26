# Adopted from https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/utils/compile.py

from contextlib import contextmanager
import os
import typing as tp

import torch
from torch import cuda

_in_cuda_graph = False
_disable_cuda_graph = False


def in_cuda_graph() -> bool:
    """Indicate whether we are in a function that is CUDA Graphed (or will be soon)."""
    return _in_cuda_graph


@contextmanager
def _set_in_cuda_graph():
    global _in_cuda_graph
    assert not _in_cuda_graph
    _in_cuda_graph = True
    try:
        yield
    finally:
        _in_cuda_graph = False


def _is_cuda_graph_enabled() -> bool:
    if _disable_cuda_graph:
        return False
    no_cuda_graph = os.environ.get("NO_CUDA_GRAPH", "")
    if no_cuda_graph.lower() not in {"0", "no", "n", ""}:
        return False
    return True


@contextmanager
def no_cuda_graph():
    """Deactivate CUDA Graphing for all the calls in this context manager."""
    global _disable_cuda_graph
    old_value = _disable_cuda_graph
    _disable_cuda_graph = True
    try:
        yield
    finally:
        _disable_cuda_graph = old_value


class CUDAGraphed:
    """Allow simple CUDA Graphing of a function.

    Args:
        func: callable, taking any number of arguments. Its tensors arguments should
            be top level args, not nested in structures (tuples, dicts, etc). Keyword
            arguments are NOT supported for simplicity.
        warmup_steps: how many call to make normally before CUDA Graphing. In particular, this
            allows torch.compiled functions to get properly compiled.
        disabled: if True, just call the func directly, useful to quickly deactivate on CPU.
    """

    def __init__(self, func: tp.Callable, warmup_steps: int = 1, disable: bool = False):
        self.func = func
        self.warmup_steps = warmup_steps
        self.disable = disable
        self._graph: cuda.CUDAGraph | None = None
        self._output: tuple | None = None
        self._args: tuple | None = None

    def reset(self, warmup_steps: int = 0) -> None:
        """Reset the state, meaning the next call we get CUDA Graphed again. Useful if some
        shapes have changed, or external state (e.g. KVCache) has changed."""
        self.warmup_steps = warmup_steps
        self._graph = None
        self._output = None
        self._args = None

    def __call__(self, *args, **kwargs) -> tp.Any:
        if kwargs:
            raise RuntimeError("Named arguments not supported for now.")
        if self.disable or not _is_cuda_graph_enabled() or in_cuda_graph():
            return self.func(*args, **kwargs)

        def _clone_tensors(args: tuple) -> tuple:
            out: list = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg = arg.clone()
                out.append(arg)
            return tuple(out)

        def _match_values_copy_tensors(args: tuple, target_args: tuple) -> None:
            if len(args) != len(target_args):
                raise ValueError(
                    f"Expected {len(target_args)}, but got {args} for CUDA Graphed function."
                )
            for idx, (source, target) in enumerate(zip(args, target_args)):
                if isinstance(target, torch.Tensor):
                    if not isinstance(source, torch.Tensor):
                        raise ValueError(
                            f"Argument #{idx} was a tensor, and is no longer (now {source})."
                        )
                    if source.shape != target.shape:
                        raise ValueError(
                            f"Argument #{idx} had shape {target.shape}, but got shape {source.shape}"
                        )
                    target.copy_(source)
                else:
                    if isinstance(source, torch.Tensor):
                        raise ValueError(
                            f"Argument #{idx} was not a tensor {target}, but is now one."
                        )
                    if source is not target and source != target:
                        raise ValueError(
                            f"Argument #{idx} changed value from {target} to {source}."
                        )

        with _set_in_cuda_graph():
            # Prevent any one under us to try and CUDA Graph things.
            if self._graph is None:
                if self.warmup_steps <= 0:
                    self._graph = cuda.CUDAGraph()
                    # Making a copy just to ensure those are not used else where.
                    self._args = _clone_tensors(args)
                    with cuda.graph(self._graph):
                        self._output = self.func(*self._args)
                    # At this point nothing really happened, so we have to make it run for real.
                    self._graph.replay()
                    return self._output
                else:
                    self.warmup_steps -= 1
                    return self.func(*args)
            else:
                assert self._args is not None
                assert self._output is not None
                _match_values_copy_tensors(args, self._args)
                self._graph.replay()
                return self._output
