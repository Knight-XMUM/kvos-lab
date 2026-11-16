"""Measure, on a Mac, what commit_sim.py predicts.

Allocates a KV pool split into Metal-visible buffers (per-layer chunks), replays
the block IDs that vLLM's FIFO (or lowest-ID-first) order would hand out for a
stream of requests, reads each newly allocated block once on the GPU, and
counts resident pages with mincore. Prints measured and predicted GiB side by
side at the same checkpoints.

No wired limit is set, so this isolates "a GPU read commits the whole buffer"
(cause 2 in vllm-metal #1198), which is what #1351 leaves.

usage (inside vllm-metal's .venv-vllm-metal):
  python sim/chunk_measure.py --chunk-mib 64 --order fifo
  python sim/chunk_measure.py --chunk-mib 0  --order fifo     # 0 = one buffer per layer
"""

from __future__ import annotations

import argparse
import ctypes
import math
import random
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import torch

from vllm_metal.tensor_bridge import torch_to_mlx

sys.path.insert(0, str(Path(__file__).parent))
from commit_sim import Fifo, Lowest  # noqa: E402

GIB = 1024**3
MIB = 1024**2
PAGE = resource.getpagesize()
libc = ctypes.CDLL(None, use_errno=True)


def resident(addr: int, nbytes: int) -> int:
    start = addr - addr % PAGE
    pages = (addr + nbytes - start + PAGE - 1) // PAGE
    vec = (ctypes.c_char * pages)()
    if libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(pages * PAGE), vec):
        raise OSError(ctypes.get_errno(), "mincore")
    return sum(1 for b in vec.raw if b & 1) * PAGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-gib", type=float, default=4)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--chunk-mib", type=int, default=64, help="0 = one buffer per layer")
    ap.add_argument("--order", choices=["fifo", "lowest"], default="fifo")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--prompt-min", type=int, default=128)
    ap.add_argument("--prompt-max", type=int, default=2048)
    ap.add_argument("--out-min", type=int, default=128)
    ap.add_argument("--out-max", type=int, default=1024)
    ap.add_argument("--requests", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.pool_gib > 8:
        sys.exit("pool above 8 GiB refused on this machine")
    marks = {1, 5, 10, 20, 40, 80, 160}

    page = 2 * args.block_size * 8 * 128 * 2  # K+V, 8 kv heads, head 128, fp16
    n_blocks = int(args.pool_gib * GIB) // (args.layers * page)
    region = n_blocks * page
    per = n_blocks if args.chunk_mib == 0 else max(1, args.chunk_mib * MIB // page)
    n_chunks = math.ceil(n_blocks / per)

    # buffers[layer][chunk]: uint8 rows of one block each, not zero-filled
    tensors, buffers = [], []
    for _ in range(args.layers):
        row_t, row_b = [], []
        for c in range(n_chunks):
            rows = min(per, n_blocks - c * per)
            t = torch.empty((rows, page), dtype=torch.uint8, device="cpu")
            row_t.append(t)
            row_b.append(torch_to_mlx(t))
        tensors.append(row_t)
        buffers.append(row_b)
    print(f"pool={args.pool_gib} GiB layers={args.layers} blocks={n_blocks} "
          f"chunk={'layer' if args.chunk_mib == 0 else str(args.chunk_mib) + 'MiB'} "
          f"({n_chunks}/layer, {args.layers * n_chunks} buffers) order={args.order}",
          flush=True)

    def measured() -> float:
        return sum(resident(t.data_ptr(), t.numel()) for lt in tensors for t in lt) / GIB

    touched: set[int] = set()

    def touch(ids: list[int]) -> None:
        by_chunk: dict[int, list[int]] = {}
        for b in ids:
            by_chunk.setdefault(b // per, []).append(b % per)
        outs = []
        for c, rows in by_chunk.items():
            idx = mx.array(rows, dtype=mx.int32)
            for layer in range(args.layers):
                outs.append(buffers[layer][c][idx].sum())
        mx.eval(*outs)
        touched.update(ids)

    def predicted() -> float:
        chunks = {b // per for b in touched}
        full = sum(min(per, n_blocks - c * per) for c in chunks) * page
        return args.layers * min(full, region) / GIB

    time.sleep(1)
    print(f"{'after allocation':<22} measured={measured():6.2f}  predicted=  0.00  (GiB)", flush=True)

    rng = random.Random(args.seed)
    pool = Fifo(n_blocks) if args.order == "fifo" else Lowest(n_blocks)
    touch([0])  # null block
    running, done = [], 0
    while done < args.requests:
        while len(running) < args.max_seqs:
            p = rng.randint(args.prompt_min, args.prompt_max)
            o = rng.randint(args.out_min, args.out_max)
            need = math.ceil(p / args.block_size)
            if len(pool) < need + 1:
                break
            ids = [pool.take() for _ in range(need)]
            touch(ids)
            running.append({"len": p, "left": o, "ids": ids})
        new = []
        for r in list(running):
            r["len"] += 1
            r["left"] -= 1
            if r["len"] % args.block_size == 1 and r["left"] > 0:
                b = pool.take()
                r["ids"].append(b)
                new.append(b)
            if r["left"] == 0:
                pool.give(r["ids"])
                running.remove(r)
                done += 1
                if done in marks:
                    if new:
                        touch(new)
                        new = []
                    time.sleep(0.5)
                    print(f"{'requests done ' + str(done):<22} measured={measured():6.2f}  "
                          f"predicted={predicted():6.2f}  (GiB)", flush=True)
        if new:
            touch(new)


if __name__ == "__main__":
    main()
