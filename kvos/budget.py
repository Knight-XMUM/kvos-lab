"""KV Cache budget & guardrail calculator for Apple Silicon Unified Memory.

Unlike discrete-GPU CUDA servers where VRAM is dedicated to the inference process,
an Apple Silicon Mac shares one physical RAM pool across:
  - macOS kernel + desktop apps + baseline wired memory
  - Model weights (pinned/resident)
  - Metal transient activation / logit buffers
  - KV Cache pool (often locked when `mx.set_wired_limit` is enabled)

This module computes conservative, swap-avoiding KV pool budgets and highlights
when `wired_limit` or uncapped MLX buffer caching risks triggering XNU memory
compression / pageouts on 16–24 GiB machines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class KVBudgetPlan:
    total_ram_gib: float
    os_and_app_headroom_gib: float
    model_weights_gib: float
    activation_scratch_gib: float
    safe_kv_budget_gib: float
    bytes_per_token_slot: int
    max_safe_kv_tokens: int
    recommended_mlx_cache_limit_gb: float
    wired_risk_level: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_os_headroom_gib(total_ram_gib: float, current_wired_gib: float | None = None) -> float:
    """Estimate non-reclaimable OS + desktop headroom on macOS."""
    base = max(4.5, total_ram_gib * 0.22)
    if current_wired_gib is not None and current_wired_gib > 0:
        # Preserve at least current baseline wired + 2.5 GiB for macOS/desktop apps
        base = max(base, current_wired_gib + 2.5)
    return round(base, 2)


def compute_kv_budget(
    *,
    total_ram_gib: float,
    model_weights_gib: float,
    num_layers: int,
    n_kv_heads: int,
    head_dim: int,
    kv_dtype_bytes: int = 2,
    os_headroom_gib: float | None = None,
    activation_scratch_gib: float = 1.5,
    current_wired_gib: float | None = None,
) -> KVBudgetPlan:
    """Calculate safe KV cache token budget and Apple Silicon guardrails."""
    if total_ram_gib <= 0:
        raise ValueError("total_ram_gib must be > 0")
    if num_layers <= 0 or n_kv_heads <= 0 or head_dim <= 0 or kv_dtype_bytes <= 0:
        raise ValueError("model KV dimensions and dtype bytes must be > 0")

    headroom = (
        round(os_headroom_gib, 2)
        if os_headroom_gib is not None
        else estimate_os_headroom_gib(total_ram_gib, current_wired_gib)
    )
    safe_kv_gib = max(0.0, total_ram_gib - headroom - model_weights_gib - activation_scratch_gib)
    bytes_per_slot = 2 * num_layers * n_kv_heads * head_dim * kv_dtype_bytes
    max_tokens = int((safe_kv_gib * (1024**3)) // bytes_per_slot)

    # Cap MLX's recycled buffer cache so old/cleared KV buffers do not stay wired
    rec_cache_limit = round(max(1.0, min(4.0, safe_kv_gib * 0.35)), 1)

    notes: list[str] = []
    if safe_kv_gib < 1.0:
        wired_risk = "HIGH"
        notes.append(
            "Safe KV budget is < 1.0 GiB; enabling mx.set_wired_limit risks XNU compression/swap."
        )
    elif total_ram_gib <= 24.5 and safe_kv_gib < 6.0:
        wired_risk = "MEDIUM"
        notes.append(
            "On 16–24 GiB unified memory, keep SGLANG_MLX_CACHE_LIMIT_GB capped and avoid FIFO full-pool touch."
        )
    else:
        wired_risk = "LOW"
        notes.append(
            "Sufficient headroom; prefer lowest-ID-first block reuse with chunked buffers to keep mincore low."
        )

    if total_ram_gib <= 24.5:
        notes.append(
            "Practical 24 GiB Mac ceiling: keep total wired (baseline + model + KV) below ~16–18 GiB."
        )

    return KVBudgetPlan(
        total_ram_gib=round(total_ram_gib, 2),
        os_and_app_headroom_gib=headroom,
        model_weights_gib=round(model_weights_gib, 2),
        activation_scratch_gib=round(activation_scratch_gib, 2),
        safe_kv_budget_gib=round(safe_kv_gib, 2),
        bytes_per_token_slot=bytes_per_slot,
        max_safe_kv_tokens=max_tokens,
        recommended_mlx_cache_limit_gb=rec_cache_limit,
        wired_risk_level=wired_risk,
        notes=notes,
    )
