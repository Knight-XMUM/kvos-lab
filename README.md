# kvos-lab

KVOS 的实验仓库：把操作系统管理内存的思路用到 LLM 推理的 KV cache 上，先从 Apple Silicon（MLX / Metal）上的真实问题做起。

## 实验 1：拆小 KV 缓冲区，能省多少常驻内存？

背景：[vllm-metal #1198](https://github.com/vllm-project/vllm-metal/issues/1198)。KV 池在服务就绪前就整体常驻，有两个原因：

1. MLX 的 wired limit 把它拥有的缓冲区锁在内存里。[#1351](https://github.com/vllm-project/vllm-metal/pull/1351) 已修。
2. GPU 读一个块，会把它所在的整个缓冲区提交成常驻；而每层只有一个大缓冲区。计划的第 2 部分是把每层拆成小缓冲区（比如 64 MiB），作者在问"按层拆还是按块范围拆"。

这个实验用模拟回答一个前置问题：**按 vLLM 现在的块分配顺序，拆小之后，长期会提交多少内存？**

### 模型

- vLLM 的 `FreeKVCacheBlockQueue` 从队头取块，释放的块放回队尾（`vllm/v1/core/kv_cache_utils.py`、`block_pool.py`），所以块号会在整个池里循环。块 0 是 null block。
- 每步前向都会读这个请求在所有层的块，所以每层碰到的块号相同。
- 缓冲区一旦被碰过就一直常驻，不会被回收。
- 不考虑前缀缓存命中。
- 对比两种分配顺序：`fifo`（vLLM 现状）和 `lowest`（总是给最小的空闲块号）。

### 结果（8 GiB 池，28 层，`--max-seqs 4`，3 个随机种子取平均；峰值实际 KV 用量约 1.02 GiB）

已提交内存（GiB），按完成的请求数：

| 完成请求数 | 每层 1 个缓冲区 | 按层拆 64 MiB | 按层拆 16 MiB | 按块范围拆 64 MiB |
|---:|---:|---:|---:|---:|
| fifo，1 | 8.00 | 1.75 | 0.88 | 0.66 |
| fifo，10 | 8.00 | 3.50 | 2.62 | 2.36 |
| fifo，20 | 8.00 | 5.25 | 4.52 | 4.39 |
| fifo，40 | 8.00 | 8.00 | 8.00 | 7.98 |
| lowest，400 | 8.00 | 1.75 | 1.31 | 1.05 |

另外两组（`sim/result-12g-busy.json`、`sim/result-4g-light.json`）结论相同。即使一次只跑一个请求、实际只用 0.08 GiB，fifo 下也在约 80 个请求后整池提交。

### 真机验证（MacBook Air M4 24GB，2026-11-16）

`sim/chunk_measure.py` 在 Mac 上真的分配拆好的缓冲区（不设 wired limit），按模拟出的块号顺序在 GPU 上读，用 `mincore` 数常驻页。4 GiB 池、28 层、80 个请求，6 种配置、42 个检查点，**实测和预测全部一致**（误差 0.00 GiB）。原始输出在 `results/`。

| 配置 | 1 个请求后 | 10 个 | 20 个 | 80 个 |
|---|---:|---:|---:|---:|
| fifo，每层 1 个缓冲区 | 4.00 | 4.00 | 4.00 | 4.00 |
| fifo，64 MiB | 1.75 | 3.50 | 4.00 | 4.00 |
| fifo，16 MiB | 0.88 | 2.62 | 4.00 | 4.00 |
| lowest，64 MiB | 1.75 | 1.75 | 1.75 | 1.75 |
| lowest，16 MiB | 0.88 | 0.88 | 1.31 | 1.31 |

运行：`bash sim/run_measure.sh`（需要 vllm-metal 的源码环境）。

### 结论

1. **在 fifo 顺序下，拆多小都一样**：块号走完一圈后（约 `池的块数 / 每个请求平均块数` 个请求），整个池都被碰过，全部常驻。拆小只推迟了这一刻。
2. **配合"优先复用最小块号"时，拆小才有长期效果**：已提交内存贴近实际峰值用量，加上一个缓冲区大小的取整。
3. **同样的缓冲区大小，按块范围拆比按层拆的下限更低**：按层拆时每层至少提交一个缓冲区，下限是 `层数 × 缓冲区大小`（28 × 64 MiB = 1.75 GiB）；按块范围拆的一个缓冲区覆盖所有层的一段块，粒度细了一个"层数"的倍数。

### 局限

- 真机验证是独立脚本，不是在 vllm serve 里测的；只测了 4 GiB 池、一台机器、每个配置一次。
- 没模拟前缀缓存、抢占、swap，也没有模拟 Metal 是否会回收长时间不用的页。
- 没考虑 kernel 里多一次查表的速度代价。

### 运行

```bash
python3 sim/commit_sim.py                     # 默认：8 GiB 池、28 层、4 并发
python3 sim/commit_sim.py --pool-gib 12 --max-seqs 8 --prompt-max 4096
python3 sim/commit_sim.py --help
```

只用 Python 标准库，不需要 GPU。

## 相关测量

- vllm-metal #1198 的复现（M4 Air 24GB，4–12 GiB 池）：[评论](https://github.com/vllm-project/vllm-metal/issues/1198#issuecomment-6888153102)
- PR #1351 的测试（main 与 PR 分支对比）：[评论](https://github.com/vllm-project/vllm-metal/pull/1351#issuecomment-6889187861)
- [`sim/`](sim/) & [`experiments/`](experiments/)：跨引擎（`vllm-metal`、`sglang`）的可复现实验与真机测量脚本。

## 实验 2：SGLang `MlxAttentionKVPool` 的惰性分配陷阱与 Wired 幽灵锁页

背景：SGLang Apple Silicon 后端中的 `MlxAttentionKVPool`（`python/sglang/srt/hardware_backend/mlx/kv_cache/mlx_kv_pool.py`）使用 `mx.zeros((pool_size, n_kv_heads, head_dim))` 创建和清理（`clear()`）每层 K/V 缓冲区，且 `MlxModelRunner` 默认开启 `mx.set_wired_limit(max_recommended_working_set_size)`。

### 真机实测（MacBook Air M4 24GB，2.00 GiB KV 池，28 层，56 个 buffer，2026-11-17）

脚本：[`experiments/sglang_kv_pool_residency.py`](experiments/sglang_kv_pool_residency.py)，原始输出：[`results/exp2-sglang-kv-pool-mac.txt`](results/exp2-sglang-kv-pool-mac.txt)。

| 阶段 | `nowired`：`mincore` | `nowired`：`mlx_active` | `nowired`：OS `wired` | `wired`（SGLang 默认）：`mincore` | `wired`：`mlx_active` | `wired`：OS `wired` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. `after __init__ (lazy)` | 0.00 GiB | +0.00 GiB | +0.00 GiB | 0.00 GiB | +0.00 GiB | +0.00 GiB |
| 2. `after mx.eval(pool)` | 2.00 GiB | +2.00 GiB | +1.93 GiB | 2.00 GiB | +2.00 GiB | +2.22 GiB |
| 3. `after unrelated op` | 2.00 GiB | +2.00 GiB | -0.10 GiB | 2.00 GiB | +2.00 GiB | +2.23 GiB |
| 4. **`after clear() (lazy)`** | **0.00 GiB** | **+0.00 GiB** | **-0.09 GiB** | **0.00 GiB** | **+0.00 GiB** | **+2.23 GiB** |
| 5. `after clear() + eval` | 2.00 GiB | +2.00 GiB | +1.91 GiB | 2.00 GiB | +2.00 GiB | +2.23 GiB |

### 核心发现

1. **启动期惰性假象（Lazy Init Trap）**：`MlxAttentionKVPool.__init__` 仅创建未求值的 `mx.zeros` 图节点，此时 `mx.get_active_memory()` 与 `mincore` 均为 `0.00 GiB`；直到首次 `eval` 或首个请求写入时才瞬间触发 2.00 GiB 物理页分配与锁页。
2. **`clear()` 后的幽灵锁页（Ghost Wired Pages）**：在 SGLang 默认的 `wired` 模式下，调用 `pool.clear()` 重新赋值为新的惰性 `mx.zeros` 后，**`mx.get_active_memory()` 降为 `0.00 GiB`，但 macOS 内核的 `wired` 内存仍死锁在 `+2.23 GiB`**（旧 buffer 退入 MLX Metal cache 且仍受 wired limit 锁定），若不配合 `mx.clear_cache()` 或受控 wired 策略，这部分物理内存无法被系统回收。
