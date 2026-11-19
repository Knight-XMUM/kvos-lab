"""Apple Silicon hardware & XNU unified-memory telemetry."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from kvos.probe import GIB, vm_stat_metrics


def _sysctl_str(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _sysctl_int(key: str) -> int | None:
    val = _sysctl_str(key)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


@dataclass
class AppleSiliconInfo:
    chip: str
    model: str
    total_ram_gib: float
    iogpu_wired_limit_mb: int | None
    mlx_recommended_working_set_gib: float | None
    vm_wired_gib: float
    vm_active_gib: float
    vm_compressed_gib: float
    vm_pageouts_gib: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_apple_silicon_info() -> AppleSiliconInfo:
    """Collect Apple Silicon hardware, IOGPU wired limit, MLX working set, and vm_stat."""
    chip = _sysctl_str("machdep.cpu.brand_string") or "Apple Silicon"
    model = _sysctl_str("hw.model") or "Mac"
    memsize = _sysctl_int("hw.memsize") or 0
    iogpu_wired_mb = _sysctl_int("iogpu.wired_limit_mb")

    mlx_rec_gib: float | None = None
    try:
        import mlx.core as mx

        info = mx.device_info()
        rec = int(info.get("max_recommended_working_set_size", 0))
        if rec > 0:
            mlx_rec_gib = round(rec / GIB, 2)
    except Exception:
        pass

    vm = vm_stat_metrics()
    return AppleSiliconInfo(
        chip=chip,
        model=model,
        total_ram_gib=round(memsize / GIB, 2),
        iogpu_wired_limit_mb=iogpu_wired_mb,
        mlx_recommended_working_set_gib=mlx_rec_gib,
        vm_wired_gib=round(vm["wired_gib"], 2),
        vm_active_gib=round(vm["active_gib"], 2),
        vm_compressed_gib=round(vm["compressed_gib"], 2),
        vm_pageouts_gib=round(vm["pageouts_gib"], 2),
    )
