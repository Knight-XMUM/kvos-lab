"""How much of a chunked KV pool gets committed when prefix caching is on.

Follow-up to commit_sim.py (vllm-metal #1198 / #1384). With prefix caching on,
vLLM keeps freed blocks that hold a full, hashed block of tokens at the back of
the free queue so later requests can hit them (LRU eviction). New blocks come
from the front, so a request uses never-touched blocks before it evicts a cached
one, and the touched block IDs spread over the whole pool. This script measures
that spread together with the prefix-cache hit rate it buys.

Model (vLLM 0.34.0, vllm/v1/core: block_pool.py, kv_cache_manager.py,
single_type_kv_cache_manager.py)
- The free queue starts as block IDs 1..N-1 in order; block 0 is the null block.
- A new request hits the longest run of its full prompt blocks whose hashes are
  cached, at most (prompt_len - 1) // block_size blocks. Hit blocks that are
  free leave the free queue (BlockPool.touch).
- Other blocks come from the front of the free queue. A block that still holds
  a hash is evicted from the cache when it is reused.
- Every block that becomes full is cached, prompt and generated tokens alike.
  Hashes chain over all earlier tokens, so a block is shared only by requests
  with the same tokens up to and including it.
- A finished request frees its blocks last block first. Blocks with a hash go
  to the back of the queue in that order; blocks without one (the partial last
  block) go to the front.
- A touched block stays committed (with prefix caching on, #1384 never gives
  chunks back).

Workload: users hold multi-turn conversations. A conversation starts with one
of a few shared system prompts (Zipf popularity), and each turn's prompt is the
whole conversation so far plus a new user message. A user thinks for a random
number of steps between turns. Prefill takes no time; each running request then
generates one token per step.

Policies:
- vllm: the queue above.
- lifo-all: every freed block goes to the front, hashed or not.
- budget-<G>: like vllm, but a never-touched block is taken only while less
  than G GiB of blocks have been touched. Past that, the least recently freed
  cached block is evicted first, and the pool grows only when no free block is
  left. budget-0 keeps the pool near the live peak; an unbounded budget is the
  vllm policy.
- age-<T>: like vllm, but when the least recently freed cached block has been
  free for more than T steps, it is evicted before a never-touched block is
  taken. The pool then grows only for blocks freed within the last T steps.
  An unbounded age is the vllm policy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict, deque

from commit_sim import GIB, MIB, committed_bytes


class FreeQueue:
    """vLLM's free block queue: pop the front, prepend, append, remove any."""

    def __init__(self) -> None:
        self.q: OrderedDict[int, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self.q)

    def first(self) -> int:
        return next(iter(self.q))

    def popleft(self) -> int:
        return self.q.popitem(last=False)[0]

    def append(self, b: int) -> None:
        self.q[b] = None

    def prepend_n(self, blocks: list[int]) -> None:
        for b in reversed(blocks):  # blocks[0] ends up at the very front
            self.q[b] = None
            self.q.move_to_end(b, last=False)

    def remove(self, b: int) -> None:
        del self.q[b]


class Pool:
    """Block pool with prefix caching. Never-touched blocks are handed out in ID
    order, so the touched IDs are always 0..next_fresh-1 and the free queue only
    holds touched blocks: [unhashed, most recent first] + [hashed, LRU first].
    vLLM's single queue keeps the never-touched blocks between those two parts.
    """

    def __init__(self, n_blocks: int, policy: str, limit: int | None):
        self.n = n_blocks
        self.policy = policy
        # "budget": touched-block limit; "age": idle steps; None = unbounded
        self.limit = limit
        self.next_fresh = 1  # block 0 is the null block
        self.freeq = FreeQueue()
        self.ref: dict[int, int] = {}
        self.hash_of: dict[int, tuple] = {}
        self.cached: dict[tuple, dict[int, None]] = {}
        self.freed_at: dict[int, int] = {}

    def num_free(self) -> int:
        return len(self.freeq) + self.n - self.next_fresh

    def lookup(self, key: tuple) -> int | None:
        blocks = self.cached.get(key)
        return next(iter(blocks)) if blocks else None

    def touch(self, b: int) -> None:
        if self.ref.get(b, 0) == 0:
            self.freeq.remove(b)
        self.ref[b] = self.ref.get(b, 0) + 1

    def take(self, now: int) -> int:
        q, fresh_left = self.freeq, self.next_fresh < self.n
        if self.policy == "lifo-all":
            use_queue = len(q) > 0
        elif q and q.first() not in self.hash_of:
            use_queue = True  # an unhashed freed block costs nothing to reuse
        elif (self.policy == "age" and q and self.limit is not None
              and now - self.freed_at[q.first()] > self.limit):
            use_queue = True  # the LRU cached block has gone cold
        elif fresh_left and (self.policy != "budget" or self.limit is None
                             or self.next_fresh - 1 < self.limit):
            use_queue = False
        else:
            use_queue = len(q) > 0
        if use_queue:
            b = q.popleft()
            self._evict(b)
        elif fresh_left:
            b = self.next_fresh
            self.next_fresh += 1
        else:
            raise RuntimeError("out of KV blocks")
        self.ref[b] = 1
        return b

    def cache(self, b: int, key: tuple) -> None:
        self.hash_of[b] = key
        self.cached.setdefault(key, {})[b] = None

    def free(self, blocks: list[int], now: int) -> None:
        front, back = [], []
        for b in reversed(blocks):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                if self.policy == "lifo-all" or b not in self.hash_of:
                    front.append(b)
                else:
                    back.append(b)
        self.freeq.prepend_n(front)
        for b in back:
            self.freeq.append(b)
            self.freed_at[b] = now

    def _evict(self, b: int) -> None:
        key = self.hash_of.pop(b, None)
        if key is not None:
            del self.cached[key][b]
            if not self.cached[key]:
                del self.cached[key]


def simulate(args, policy: str, limit: float | None, seed: int):
    """limit: GiB for "budget", idle steps for "age", unused otherwise."""
    bs = args.block_size
    page = 2 * bs * args.kv_heads * args.head_dim * 2  # K+V, fp16, one layer
    n_blocks = int(args.pool_gib * GIB) // (args.layers * page)
    if policy == "budget" and limit is not None:
        limit = int(limit * GIB) // (args.layers * page)
    pool = Pool(n_blocks, policy, None if limit is None else int(limit))
    rng = random.Random(seed)
    sys_lens = [rng.randint(args.sys_min, args.sys_max) for _ in range(args.sys_prompts)]
    sys_weights = [1 / (i + 1) for i in range(args.sys_prompts)]
    conv_ids = iter(range(1, 1 << 62))

    def new_conv() -> dict:
        s = rng.choices(range(args.sys_prompts), sys_weights)[0] if args.sys_prompts else None
        n = 0 if s is None else sys_lens[s]
        return {"id": next(conv_ids), "sys": s, "sys_len": n, "len": n,
                "turns_left": rng.randint(1, args.max_turns)}

    def key(conv: dict, b: int) -> tuple:
        if conv["sys"] is not None and (b + 1) * bs <= conv["sys_len"]:
            return ("sys", conv["sys"], b)  # same tokens in every conversation
        return ("conv", conv["id"], b)

    users = [{"conv": new_conv(), "ready": rng.randint(0, args.think_max), "busy": False}
             for _ in range(args.users)]
    waiting: deque = deque()
    running: list[dict] = []
    now = done = peak_live = 0
    per_req: list[tuple[int, int]] = []  # (prompt tokens, hit tokens)
    snaps = []
    while done < args.requests:
        for u in users:
            if not u["busy"] and u["ready"] <= now:
                u["busy"] = True
                u["prompt"] = u["conv"]["len"] + rng.randint(args.user_min, args.user_max)
                u["out"] = rng.randint(args.out_min, args.out_max)
                waiting.append(u)
        while waiting and len(running) < args.max_seqs:
            u = waiting[0]
            conv, n_prompt = u["conv"], u["prompt"]
            hits = []
            for b in range((n_prompt - 1) // bs):
                blk = pool.lookup(key(conv, b))
                if blk is None:
                    break
                hits.append(blk)
            need = -(-n_prompt // bs) - len(hits)
            free_hits = sum(1 for blk in hits if pool.ref.get(blk, 0) == 0)
            if pool.num_free() < need + free_hits + 1:
                break
            waiting.popleft()
            for blk in hits:
                pool.touch(blk)
            blocks = hits + [pool.take(now) for _ in range(need)]
            for b in range(len(hits), n_prompt // bs):
                pool.cache(blocks[b], key(conv, b))
            running.append({"user": u, "conv": conv, "len": n_prompt, "left": u["out"],
                            "blocks": blocks, "prompt": n_prompt, "hit": len(hits) * bs})
        if not running:
            if waiting:
                raise RuntimeError("pool too small for a single request")
            now = min(u["ready"] for u in users)
            continue
        peak_live = max(peak_live, sum(len(r["blocks"]) for r in running))
        for r in list(running):
            if r["len"] % bs == 0:
                r["blocks"].append(pool.take(now))
            r["len"] += 1
            r["left"] -= 1
            if r["len"] % bs == 0:
                b = r["len"] // bs - 1
                pool.cache(r["blocks"][b], key(r["conv"], b))
            if r["left"] > 0:
                continue
            running.remove(r)
            pool.free(r["blocks"], now)
            conv, u = r["conv"], r["user"]
            conv["len"] = r["len"]
            conv["turns_left"] -= 1
            if conv["turns_left"] == 0:
                u["conv"] = new_conv()
            u["busy"] = False
            u["ready"] = now + 1 + rng.randint(0, args.think_max)
            per_req.append((r["prompt"], r["hit"]))
            done += 1
            if done in args.marks:
                hit = sum(h for _, h in per_req) / sum(p for p, _ in per_req)
                snaps.append((done, pool.next_fresh, hit))
            if done == args.requests:
                break
        now += 1
    tail = per_req[len(per_req) // 2:]
    steady_hit = sum(h for _, h in tail) / sum(p for p, _ in tail)
    return page, n_blocks, peak_live, snaps, steady_hit


def parse_limit(s: str) -> float | None:
    return None if s == "inf" else float(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-gib", type=float, default=8)
    ap.add_argument("--layers", type=int, default=28)  # Qwen3-0.6B
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--users", type=int, default=16)
    ap.add_argument("--sys-prompts", type=int, default=4)
    ap.add_argument("--sys-min", type=int, default=256)
    ap.add_argument("--sys-max", type=int, default=1024)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--user-min", type=int, default=32)
    ap.add_argument("--user-max", type=int, default=512)
    ap.add_argument("--out-min", type=int, default=64)
    ap.add_argument("--out-max", type=int, default=512)
    ap.add_argument("--think-max", type=int, default=2000, help="steps between turns")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", default="0,2,4", help="GiB, comma-separated; inf allowed")
    ap.add_argument("--ages", default="1000,2000,4000,8000",
                    help="idle steps, comma-separated; inf allowed")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    args.marks = {m for m in (1, 10, 50, 100, 200, 400, 1000, args.requests) if m <= args.requests}
    configs = [("per-layer", None), ("per-layer", 64 * MIB), ("per-layer", 16 * MIB),
               ("per-block-range", 64 * MIB)]
    policies = ([("vllm", "vllm", None), ("lifo-all", "lifo-all", None)]
                + [(f"budget-{b}", "budget", parse_limit(b)) for b in args.budgets.split(",") if b]
                + [(f"age-{a}", "age", parse_limit(a)) for a in args.ages.split(",") if a])
    out = {"args": {k: v for k, v in vars(args).items() if k != "marks"}, "rows": []}
    summary = []
    for name, policy, budget in policies:
        acc: dict = {}
        peaks, steadies = [], []
        for s in range(args.seeds):
            page, n, peak, snaps, steady = simulate(args, policy, budget, s)
            peaks.append(peak)
            steadies.append(steady)
            for done, touched, hit in snaps:
                ids = set(range(touched))
                acc.setdefault(("hit", done), []).append(hit)
                acc.setdefault(("touched", done), []).append(touched * args.layers * page / GIB)
                for layout, c in configs:
                    acc.setdefault((layout, c, done), []).append(
                        committed_bytes(ids, page, args.layers, n, layout, c) / GIB)
        live_gib = max(peaks) * args.layers * page / GIB
        steady = sum(steadies) / len(steadies)
        print(f"\n== {name}  pool={args.pool_gib} GiB  blocks={n}  "
              f"peak live KV={live_gib:.2f} GiB  hit rate (2nd half)={steady:.1%}")
        print("requests done | one buffer/layer | per-layer 64MiB | per-layer 16MiB "
              "| per-block 64MiB | hit rate so far")
        for done in sorted(args.marks):
            vals = [acc.get((layout, c, done)) for layout, c in configs]
            hits = acc.get(("hit", done))
            cells = [f"{sum(v) / len(v):6.2f}" if v else "     -" for v in vals]
            hit = f"{sum(hits) / len(hits):6.1%}" if hits else "     -"
            print(f"{done:>13} | " + " | ".join(f"{x:>15}" for x in cells) + f" | {hit:>15}")
            for (layout, c), v in zip(configs, vals):
                out["rows"].append({"policy": name, "layout": layout,
                                    "chunk_mib": None if c is None else c // MIB,
                                    "requests_done": done,
                                    "committed_gib": None if not v else round(sum(v) / len(v), 3),
                                    "hit_rate": None if not hits else round(sum(hits) / len(hits), 4)})
        ends = [acc.get(k) for k in (("touched", args.requests),
                                     ("per-layer", 64 * MIB, args.requests),
                                     ("per-block-range", 64 * MIB, args.requests))]
        summary.append((name, [sum(v) / len(v) for v in ends], steady, live_gib))
    print("\n== summary after", args.requests, "requests (mean of", args.seeds, "seeds), GiB")
    print("policy       | touched | per-layer 64MiB | per-block 64MiB | hit rate (2nd half) "
          "| peak live KV")
    for name, (touched, layer64, block64), steady, live in summary:
        print(f"{name:<12} | {touched:7.2f} | {layer64:15.2f} | {block64:15.2f} | "
              f"{steady:19.1%} | {live:12.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
