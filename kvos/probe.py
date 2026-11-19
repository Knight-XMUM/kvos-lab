"""Apple Silicon Unified Memory & KV Residency Probe (mincore + vm_stat + MLX).

Why this exists:
On Apple Silicon, CPU and GPU share one physical memory pool managed by XNU (macOS)
with 16 KiB pages, dynamic memory compression, and Metal IOGPU wired limits.
Standard LLM engine metrics (`mx.get_active_memory()`, `torch.mps.current_allocated_memory()`)
only report *logical* allocator bookkeeping and miss:
  1. Lazy allocation traps: arrays created with `mx.zeros()` occupy 0 physical pages
     until evaluated (`mincore` = 0 GiB vs logical pool capacity).
  2. Metal buffer granularity: touching 1 KV block on the GPU commits the entire
     underlying Metal buffer into resident RAM.
  3. Wired memory lock-in: when `mx.set_wired_limit()` is active, calling `pool.clear()`
     re-assigns lazy arrays without evaluating them or clearing MLX's cache, leaving
     old physical pages locked (`Pages wired down` in `vm_stat`).
"""

from __future__ import annotations

import ctypes
import re
import resource
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Iterable

GIB = 1024**3
MIB = 1024**2
PAGE_SIZE = resource.getpagesize()
_libc = ctypes.CDLL(None, use_errno=True)


def mincore_bytes(addr: int, nbytes: int) -> int:
    """Count physically resident bytes in [addr, addr + nbytes) via Darwin mincore(2)."""
    if nbytes <= 0 or addr <= 0:
        return 0
    start = addr - (addr % PAGE_SIZE)
    pages = (addr + nbytes - start + PAGE_SIZE - 1) // PAGE_SIZE
    vec = (ctypes.c_char * pages)()
    if _libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(pages * PAGE_SIZE), vec):
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(1 for b in vec.raw if b & 1) * PAGE_SIZE


def buffer_address_and_nbytes(buf: Any) -> tuple[int, int]:
    """Extract unified-memory virtual address and byte size from an MLX array or PyTorch tensor.

    Zero-copy path on Apple Silicon:
    - PyTorch CPU tensor: `buf.data_ptr()` and `buf.numel() * buf.element_size()`
    - MLX array: `np.asarray(buf)` exposes a zero-copy view over unified memory
      once the array has been evaluated (`mx.eval(buf)`). Note that calling
      `np.asarray` on an *unevaluated* MLX array triggers evaluation; pass
      `evaluated=False` in high-level callers if the array is still lazy.
    """
    if hasattr(buf, "data_ptr") and hasattr(buf, "numel") and hasattr(buf, "element_size"):
        return int(buf.data_ptr()), int(buf.numel() * buf.element_size())
    import numpy as np

    view = np.asarray(buf)
    return int(view.ctypes.data), int(view.nbytes)


def mincore_gib(buffers: Iterable[Any], *, already_evaluated: bool = True) -> float:
    """Sum physical page residency (GiB) across a collection of unified-memory buffers."""
    if not already_evaluated:
        return 0.0
    total = 0
    for b in buffers:
        addr, nbytes = buffer_address_and_nbytes(b)
        total += mincore_bytes(addr, nbytes)
    return total / GIB


def vm_stat_metrics() -> dict[str, float]:
    """Read XNU virtual memory counters from `vm_stat` in GiB (wired, active, compressed, pageouts)."""
    out = subprocess.check_output(["vm_stat"], text=True)
    m_page = re.search(r"page size of (\d+) bytes", out)
    page_bytes = int(m_page.group(1)) if m_page else PAGE_SIZE

    def _pages(label: str) -> int:
        m = re.search(rf"{re.escape(label)}:\s+(\d+)\.", out)
        return int(m.group(1)) if m else 0

    return {
        "wired_gib": _pages("Pages wired down") * page_bytes / GIB,
        "active_gib": _pages("Pages active") * page_bytes / GIB,
        "compressed_gib": _pages("Pages occupied by compressor") * page_bytes / GIB,
        "pageouts_gib": _pages("Pageouts") * page_bytes / GIB,
    }


def mlx_memory_metrics() -> dict[str, float]:
    """Read MLX logical allocator metrics in GiB if MLX is available."""
    try:
        import mlx.core as mx

        return {
            "mlx_active_gib": mx.get_active_memory() / GIB,
            "mlx_cache_gib": mx.get_cache_memory() / GIB,
            "mlx_peak_gib": mx.get_peak_memory() / GIB,
        }
    except Exception:
        return {
            "mlx_active_gib": 0.0,
            "mlx_cache_gib": 0.0,
            "mlx_peak_gib": 0.0,
        }


@dataclass
class ProbeSnapshot:
    """Single point-in-time snapshot aligning MLX logical memory, mincore residency, and XNU vm_stat."""

    stage: str
    logical_pool_gib: float
    mincore_gib: float
    mlx_active_delta_gib: float
    mlx_cache_delta_gib: float
    wired_delta_gib: float
    compressed_delta_gib: float

    @property
    def ghost_wired_gib(self) -> float:
        """Physical wired memory locked above what MLX reports as active logical memory."""
        return max(0.0, self.wired_delta_gib - self.mlx_active_delta_gib)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ghost_wired_gib"] = round(self.ghost_wired_gib, 3)
        return d


class MemoryProbe:
    """Tracks Unified Memory deltas across KV pool lifecycle stages on Apple Silicon."""

    def __init__(self, logical_pool_gib: float = 0.0) -> None:
        self.logical_pool_gib = logical_pool_gib
        self._base_vm = vm_stat_metrics()
        self._base_mlx = mlx_memory_metrics()
        self.snapshots: list[ProbeSnapshot] = []

    def record(
        self,
        stage: str,
        buffers: Iterable[Any] = (),
        *,
        buffers_evaluated: bool = True,
    ) -> ProbeSnapshot:
        vm = vm_stat_metrics()
        mlx_m = mlx_memory_metrics()
        res_gib = mincore_gib(buffers, already_evaluated=buffers_evaluated) if buffers else 0.0
        snap = ProbeSnapshot(
            stage=stage,
            logical_pool_gib=self.logical_pool_gib,
            mincore_gib=res_gib,
            mlx_active_delta_gib=mlx_m["mlx_active_gib"] - self._base_mlx["mlx_active_gib"],
            mlx_cache_delta_gib=mlx_m["mlx_cache_gib"] - self._base_mlx["mlx_cache_gib"],
            wired_delta_gib=vm["wired_gib"] - self._base_vm["wired_gib"],
            compressed_delta_gib=vm["compressed_gib"] - self._base_vm["compressed_gib"],
        )
        self.snapshots.append(snap)
        return snap

    def format_table(self) -> str:
        header = (
            f"{'Stage':<24} | {'Logical':>7} | {'mincore':>7} | "
            f"{'MLX Act':>7} | {'MLX Cache':>9} | {'OS Wired':>8} | {'Ghost Lock':>10}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for s in self.snapshots:
            rows.append(
                f"{s.stage:<24} | {s.logical_pool_gib:>6.2f}G | {s.mincore_gib:>6.2f}G | "
                f"{s.mlx_active_delta_gib:>+6.2f}G | {s.mlx_cache_delta_gib:>+8.2f}G | "
                f"{s.wired_delta_gib:>+7.2f}G | {s.ghost_wired_gib:>9.2f}G"
            )
        return "\n".join(rows)
