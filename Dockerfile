# GPU PDE Lab -- a second door into the project, for people who do not run NixOS.
#
# The flake is the supported development environment and stays that way. This
# image exists so that "can I run your benchmarks" does not first require
# adopting someone else's operating system.
#
# Build:
#   docker build -t gpu-pde-lab .
#
# Headless benchmarks (the common case; needs nvidia-container-toolkit):
#   docker run --rm --gpus all gpu-pde-lab
#   docker run --rm --gpus all gpu-pde-lab python scripts/run_roofline.py --sizes 512 1024 2048
#   docker run --rm --gpus all -v "$PWD/out:/work/docs/results" gpu-pde-lab \
#       python scripts/run_roofline.py
#
# Figures, no GPU needed:
#   docker run --rm -v "$PWD/out:/work/docs/figures" gpu-pde-lab \
#       python scripts/make_figures.py
#
# Tests:
#   docker run --rm --gpus all gpu-pde-lab pytest -q
#
# The desktop GUI needs a display forwarded in, which is fiddly and
# platform-specific; see the "Desktop app" note at the bottom of this file.

# The CUDA runtime matters: torch's cu12x wheels bundle their own CUDA runtime
# but still need a matching *driver* on the host, supplied by
# nvidia-container-toolkit. The base image provides the userspace libraries the
# wheels dynamically link against.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Qt has no display in a container by default. Every test and script here
    # is headless; this keeps the widget tests from trying to open one.
    QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        # libGL and friends: pyqtgraph imports OpenGL bindings even when
        # rendering offscreen, and PySide6 links libEGL unconditionally.
        libgl1 \
        libegl1 \
        libglib2.0-0 \
        libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /work

# Dependency metadata first, so editing source does not invalidate the
# expensive torch layer below.
COPY pyproject.toml README.md ./

# Torch separately and explicitly: resolving it as a transitive dependency
# picks whatever the index defaults to, and the CUDA build has to be chosen
# deliberately. cu124 matches the base image.
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124

COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY doc.md LICENSE ./

RUN python3 -m pip install -e ".[dev]"

# Triton locates libcuda.so.1 by shelling out to `ldconfig -p`. That works on a
# normal Ubuntu base, but nvidia-container-toolkit mounts the driver libraries
# in after the image is built, so the linker cache can be stale at runtime.
# Pointing Triton at the mount directly skips the probe entirely.
ENV TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu

# Default: report what this machine can actually run, rather than starting a
# long benchmark nobody asked for.
CMD ["python3", "-c", "\
from gpu_pde_lab.core.backend import cuda_is_usable, cuda_free_memory_bytes;\
from gpu_pde_lab.kernels.registry import available_implementations;\
from gpu_pde_lab.benchmarking.roofline import collect_device_info;\
d = collect_device_info();\
print('GPU PDE Lab');\
print('  device:', d.summary() if d else 'none usable');\
print('  free  :', (f'{cuda_free_memory_bytes()/1024**2:.0f} MB' if cuda_is_usable() else 'n/a'));\
print('  implementations:');\
[print(f'    {i.name:<12} ' + ('ok' if i.available else f'unavailable -- {i.unavailable_reason}')) for i in available_implementations()];\
print();\
print('  python scripts/run_roofline.py    measure against the memory roofline');\
print('  python scripts/count_kernels.py   kernel launches and array passes');\
print('  python scripts/make_figures.py    regenerate docs/figures (no GPU needed)');\
print('  pytest -q                         the test suite')\
"]

# Desktop app
# -----------
# `python3 -m gpu_pde_lab.app` works in this image but needs a display and the
# `lab` extra (`pip install -e '.[lab]'`, already covered by `[dev]` above).
# On Linux with X11:
#
#   xhost +local:docker
#   docker run --rm --gpus all -e DISPLAY=$DISPLAY -e QT_QPA_PLATFORM=xcb \
#       -v /tmp/.X11-unix:/tmp/.X11-unix gpu-pde-lab python3 -m gpu_pde_lab.app
#
# This is unsupported and varies by host; the flake is the tested path for the
# GUI, and the headless entry points above are what this image is really for.
