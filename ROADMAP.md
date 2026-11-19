# kvos-lab 长期规划（2026-11 → 2028-03）

> 整合原 `Infra/MOONSHOT.md`（初试闸门、A/B/C 三件套、408 映射）与 `Infra/docs/research/DEFENSE.md`（实证边界），全面转向 **Apple Silicon（MLX / Metal）统一内存与 KV 常驻观测治理**，不再依赖高成本云机 GPU。

## 0. 两条硬约束

1. **初试是唯一闸门（2027-12）**：备考期间项目严格控制在 **<= 5 小时/周**；一旦 408 或数学模拟落后目标节奏，项目立即切入维护模式（<= 2 小时/周）。
2. **与 408 联动，不做无效重复**：
   - `mincore` 16 KiB 页表、`vm_stat` `wired` / `compressed` / `pageouts` ↔ **408 操作系统·虚拟内存与页面置换**
   - `FifoAllocator` vs `LowestIdAllocator`、`LRU` / `CLOCK` 双平面驱逐 ↔ **408 计组·Cache 映射与替换算法**
   - Radix 前缀共享与链式哈希失配代价 ↔ **408 数据结构·树与散列**

## 1. 仓库定位与“三件套”对齐

| 套件 | 在新体系里的落地 | 当前进度 |
| :--- | :--- | :--- |
| **A：引擎与 OS 内功** | `Infra/docs/learning/` + `Infra/labs/` 作为 408 配套源码读物；`kvos/policy.py` 提炼双平面（`live` / `context`）与 `LRU/CLOCK/LowestId` 原语 | `lab01–02` 已通；`kvos/policy.py` 已落地 |
| **B：独立研究与工具** | `kvos-lab`（`kvos status` / `kvos budget` / `MemoryProbe`）+ M4 Air 24GB 跨引擎实测数据与对比图 | `v0.1` 已可运行；Exp 1（`vllm-metal`）与 Exp 2（`sglang`）已完成真机实测 |
| **C：上游硬通货** | `vllm-metal` 实测分析 + `sglang` Apple Silicon / MLX 修复 PR | `vllm-metal` #1198 / #1351 贡献完成；SGLang PR #50939、#50963、#50964 已提交 |

## 2. 分阶段路线图

| 阶段 | 时间 | 每周投入 | 核心交付 | 验收标准 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1：底座与实证（当前）** | 2026-11 ~ 2026-12 | <= 5h | 完成 `kvos v0.1`（`probe` / `budget` / `policy`）；跟进 SGLang 3 个 PR；用 `kvos` 测出 `mlx-lm` vs `sglang` 在 24GB 上的 `mincore`/`wired`/吞吐对比 | 单元测试全过；`results/` 有一组跨引擎 CSV/图 |
| **Stage 2：24GB 拐点压测与护栏** | 2027-01 ~ 2027-04 | <= 5h | 在 M4 Air 24GB 跑梯度 KV 压测，标定 `compressor` 与 `pageouts` 导致吞吐掉崖的临界点；校准 `kvos budget` 公式 | 一张主图 + 两页技术说明定稿（复试/套磁可直接发） |
| **Stage 3：自适应 Governor** | 2027-05 ~ 2027-07 | <= 5h | 基于 `kvos/policy.py` 与 `vm_stat` 压力信号实现轻量 `KVGovernor` Hook，验证自动降 wired / 驱逐冷前缀效果 | 若稳定有效，拆小 PR 反哺 `sglang` / `vllm-metal` |
| **Stage 4：初试冲刺维护期** | 2027-08 ~ 2027-12 | <= 2h | 冻结新功能，只回 PR/Issue、保持仓库单测绿色，全力冲刺初试 | 初试过线 |
| **Stage 5：套磁与复试冲刺** | 2028-01 ~ 2028-03 | ~8–10h | 打磨 v1.0 release、90 秒/3 分钟英文与中文项目口述稿、演练 `DEFENSE` 追问清单 | 所有数据可指向 `results/` 原始输出，零夸大 |

## 3. 复试汇报边界（继承自 `Infra/docs/research/DEFENSE.md`）

- **怎么讲**：定位为“针对 24GB 消费级 Apple Silicon 统一内存，打通应用逻辑层、页表 `mincore` 层与 XNU `wired` 层的 KV 常驻观测与安全治理工具，并据此给 `vllm-metal` 和 `sglang` 提交了真实测量与修复”。
- **不踩坑**：不吹成“重造了推理引擎”；每项结论只引用 `results/` 里有原始记录的数据；明确区分“已在 M4 Air 实测”与“模拟器推演”的边界。
