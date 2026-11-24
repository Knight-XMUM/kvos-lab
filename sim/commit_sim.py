"""How much of a chunked KV pool ends up committed under vLLM's block order.

Question (from vllm-metal #1198 / #1351 part 2): if each layer's KV region is
split into smaller Metal buffers, and a GPU read commits the whole buffer it
lands in, how much memory is committed over time?

Model
- vLLM's FreeKVCacheBlockQueue hands out blocks from the head. Where freed
  blocks go depends on prefix caching (BlockPool.free_blocks in
  vllm/v1/core/block_pool.py): with caching on, blocks with a hash go back to
  the tail for LRU eviction, so block IDs walk through the whole pool; with
  caching off, every freed block goes to the front and is reused first.
  Block 0 is the null block.
- Every forward step touches the request's blocks in every layer, so all layers
  see the same block IDs.
- A buffer, once touched, stays committed (nothing decommits it).
- No prefix-cache hits (every request gets fresh blocks).

Layouts, for a buffer size C (bytes):
- per-layer chunks: each layer region split into C-byte buffers; one buffer
  covers C / page_bytes block IDs of one layer.
- per-block-range: one buffer holds a block range for all layers; it covers
  C / (layers * page_bytes) block IDs.
Allocation orders:
- fifo: vLLM with prefix caching on (every freed block treated as hashed).
- lowest: always hand out the lowest free block ID (a min-heap).
- lifo: vLLM with prefix caching off (freed blocks go to the front).
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from collections import deque

GIB = 1024**3
MIB = 1024**2


class Fifo:
    def __init__(self, n: int):
        self.q = deque(range(1, n))  # block 0 is the null block

    def take(self) -> int:
        return self.q.popleft()

    def give(self, ids: list[int]) -> None:
        self.q.extend(reversed(ids))  # vLLM frees a request's blocks reversed

    def __len__(self) -> int:
        return len(self.q)


class Lowest:
    def __init__(self, n: int):
        self.h = list(range(1, n))

    def take(self) -> int:
        return heapq.heappop(self.h)

    def give(self, ids: list[int]) -> None:
        for i in ids:
            heapq.heappush(self.h, i)

    def __len__(self) -> int:
        return len(self.h)


class Lifo:
    def __init__(self, n: int):
        self.q = deque(range(1, n))  # block 0 is the null block

    def take(self) -> int:
        return self.q.popleft()

    def give(self, ids: list[int]) -> None:
        # vLLM frees reversed(ids) and prepend_n keeps that order, so ids[-1] is next
        self.q.extendleft(ids)

    def __len__(self) -> int:
        return len(self.q)


ORDERS = {"fifo": Fifo, "lowest": Lowest, "lifo": Lifo}


def simulate(args, order: str, seed: int):
    page = 2 * args.block_size * args.kv_heads * args.head_dim * 2  # K+V, fp16
    n_blocks = int(args.pool_gib * GIB) // (args.layers * page)
    rng = random.Random(seed)
    pool = ORDERS[order](n_blocks)
    touched = {0}  # null block is read by padding slots
    running: list[dict] = []
    done = 0
    live_peak = 0
    timeline = []  # (requests_done, touched_ids snapshot size, live blocks, max id)
    snapshots = []
    while done < args.requests:
        while len(running) < args.max_seqs:
            p = rng.randint(args.prompt_min, args.prompt_max)
            o = rng.randint(args.out_min, args.out_max)
            need = math.ceil(p / args.block_size)
            if len(pool) < need + 1:
                break
            ids = [pool.take() for _ in range(need)]
            touched.update(ids)
            running.append({"len": p, "left": o, "ids": ids})
        # one decode step
        for r in list(running):
            r["len"] += 1
            r["left"] -= 1
            if r["len"] % args.block_size == 1 and r["left"] > 0:
                b = pool.take()
                r["ids"].append(b)
                touched.add(b)
            if r["left"] == 0:
                pool.give(r["ids"])
                running.remove(r)
                done += 1
                if done in args.marks:
                    snapshots.append((done, set(touched)))
        live = sum(len(r["ids"]) for r in running)
        live_peak = max(live_peak, live)
    return page, n_blocks, live_peak, snapshots


def committed_bytes(touched: set[int], page: int, layers: int, n_blocks: int,
                    layout: str, chunk: int | None) -> float:
    region = n_blocks * page
    if chunk is None:  # one buffer per layer (main / #1351 part 1)
        return layers * region if touched else 0
    if layout == "per-layer":
        per = max(1, chunk // page)
        chunks = {b // per for b in touched}
        return layers * min(len(chunks) * per * page, region)
    per = max(1, chunk // (layers * page))
    chunks = {b // per for b in touched}
    return min(len(chunks) * per * page * layers, layers * region)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-gib", type=float, default=8)
    ap.add_argument("--layers", type=int, default=28)  # Qwen3-0.6B
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--prompt-min", type=int, default=128)
    ap.add_argument("--prompt-max", type=int, default=2048)
    ap.add_argument("--out-min", type=int, default=128)
    ap.add_argument("--out-max", type=int, default=1024)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    args.marks = {1, 5, 10, 20, 40, 80, 160, 400}
    configs = [("per-layer", None), ("per-layer", 256 * MIB), ("per-layer", 64 * MIB),
               ("per-layer", 16 * MIB), ("per-block-range", 64 * MIB)]
    out = {"args": {k: v for k, v in vars(args).items() if k != "marks"}, "rows": []}
    for order in ORDERS:
        acc: dict = {}
        peaks = []
        for s in range(args.seeds):
            page, n, peak, snaps = simulate(args, order, s)
            peaks.append(peak)
            for done, t in snaps:
                for layout, c in configs:
                    key = (layout, c, done)
                    acc.setdefault(key, []).append(
                        committed_bytes(t, page, args.layers, n, layout, c) / GIB)
        live_gib = max(peaks) * args.layers * page / GIB
        print(f"\n== order={order}  pool={args.pool_gib} GiB  blocks={n}  "
              f"peak live KV={live_gib:.2f} GiB (max over seeds)")
        hdr = "requests done | " + " | ".join(
            f"{l[:9]} {('layer' if c is None else str(c // MIB) + 'MiB')}" for l, c in configs)
        print(hdr)
        for done in sorted(args.marks):
            vals = []
            for layout, c in configs:
                v = acc.get((layout, c, done))
                vals.append(f"{sum(v) / len(v):.2f}" if v else "-")
            print(f"{done:>13} | " + " | ".join(f"{x:>14}" for x in vals))
            for (layout, c), v in zip(configs, vals):
                out["rows"].append({"order": order, "layout": layout,
                                    "chunk_mib": None if c is None else c // MIB,
                                    "requests_done": done, "committed_gib": v,
                                    "peak_live_gib": round(live_gib, 3)})
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
