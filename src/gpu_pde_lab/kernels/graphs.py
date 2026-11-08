"""CUDA graph capture: the same memory traffic, none of the launch overhead.

This implementation exists to separate two bottlenecks that a wall-clock
number alone cannot tell apart.

The baseline is slow for two independent reasons. It moves about four
array-sized blocks of memory per step when one read and one write would do
(*bandwidth-bound*), and it asks the host to launch a dozen-odd kernels per
step, each with its own launch latency of a few microseconds (*launch-bound*).

A CUDA graph fixes only the second. Capture records the op sequence once, and
replay hands the whole sequence to the driver as a single submission, so the
per-launch cost is paid once instead of a dozen times. The arithmetic and the
memory traffic are byte-for-byte what the baseline does.

So the two optimized implementations answer different questions:

- ``cuda_graph`` vs ``baseline`` measures how much was launch overhead.
- ``triton`` vs ``cuda_graph`` measures how much was memory traffic.

On a small grid the first dominates -- there is barely any memory to move, and
launch latency is the entire runtime, which is why the project's own CPU-vs-GPU
sweep finds the GPU *losing* below a few hundred cells on a side. On a large
grid the second dominates and the graph buys almost nothing. Reporting both is
the honest way to show that "use the GPU better" is really two separate pieces
of advice depending on problem size.

Constraints that the capture imposes
------------------------------------
A graph replays a *fixed* sequence of operations against *fixed* memory
addresses. That means:

- The field has to live in a buffer allocated once, before capture, and be
  updated in place on replay. A step that returns a freshly allocated tensor
  cannot be captured, so this module keeps a persistent pair of buffers and
  copies the result back into the input each replay.
- Shapes, dtype, dt and alpha are all baked in at capture time. Changing any
  of them requires recapturing, which is why :func:`make_graphed_heat_step`
  takes them up front rather than per call.
- Capture must happen on a non-default stream with prior work drained, which
  ``torch.cuda.graph`` handles, after warmup iterations on a side stream so
  that lazily-initialized state (autotuned kernels, cuBLAS handles, the
  caching allocator's blocks) is not captured mid-initialization.
"""

from __future__ import annotations

from ..core.boundary import BoundaryKind
from .baseline import heat_step_baseline


def cuda_graph_is_available() -> bool:
    """True if this torch build has CUDA graphs and a usable device."""
    try:
        import torch
    except ImportError:
        return False
    if not hasattr(torch.cuda, "CUDAGraph"):
        return False
    from ..core.backend import cuda_is_usable

    return cuda_is_usable()


def make_graphed_heat_step(
    shape: tuple[int, int],
    dt: float,
    alpha: float,
    dx: float,
    dy: float,
    boundary: BoundaryKind,
    dirichlet_value: float = 0.0,
    dtype_str: str = "fp32",
    warmup_iters: int = 3,
):
    """Capture one baseline heat step into a CUDA graph.

    Returns a ``GraphedHeatStep`` whose ``step(u)`` replays the graph and
    returns the updated field. Everything about the computation is fixed at
    capture time; see the module docstring for why.
    """
    import torch

    if not hasattr(torch.cuda, "CUDAGraph"):
        raise RuntimeError("this torch build has no CUDA graph support")

    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[dtype_str]

    static_in = torch.zeros(shape, dtype=torch_dtype, device="cuda")

    # Warm up on a side stream. The first few executions allocate workspace
    # and initialize library handles; capturing that would either bake a
    # one-off allocation into the graph or fail outright.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup_iters):
            heat_step_baseline(static_in, dt, alpha, dx, dy, boundary, dirichlet_value)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = heat_step_baseline(static_in, dt, alpha, dx, dy, boundary, dirichlet_value)

    return GraphedHeatStep(graph=graph, static_in=static_in, static_out=static_out)


class GraphedHeatStep:
    """A captured heat step, replayable against its own fixed buffers.

    ``static_in`` and ``static_out`` are the exact tensors the graph was
    recorded against; their addresses are part of the recording, so they must
    be written through rather than rebound. Hence ``copy_`` everywhere below
    and no assignment.
    """

    def __init__(self, graph, static_in, static_out):
        self.graph = graph
        self.static_in = static_in
        self.static_out = static_out

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.static_in.shape)

    def step(self, u):
        """Advance ``u`` by one step, returning a view of the graph's output buffer.

        The returned tensor is overwritten by the next call. Callers that need
        to keep a result must clone it -- the same contract as reusing any
        preallocated output buffer.
        """
        self.static_in.copy_(u)
        self.graph.replay()
        return self.static_out

    def run(self, u, n_steps: int):
        """Replay ``n_steps`` times, feeding each output back as the next input.

        This is the shape the benchmark measures: the per-step host cost of a
        graph replay is what the whole implementation exists to demonstrate,
        and it only shows up across a run of steps.
        """
        self.static_in.copy_(u)
        for _ in range(n_steps):
            self.graph.replay()
            self.static_in.copy_(self.static_out)
        return self.static_in
