#!/bin/bash
# 在 Mac 终端运行：bash ~/Develop/kvos-lab/sim/run_measure.sh
# 4 GiB 池、28 层；3 种缓冲区大小 × 2 种分配顺序，各一个独立进程。约 3–5 分钟。
# 不设 wired limit，只测"读一块提交整个缓冲区"。
set -u
LAB=~/Develop/kvos-lab
OUT=$LAB/results
mkdir -p "$OUT"
source ~/Develop/oss-contrib/vllm-metal/.venv-vllm-metal/bin/activate || exit 1
cd "$LAB"
for order in fifo lowest; do
  for chunk in 0 64 16; do
    f="$OUT/measure-${order}-chunk${chunk}.txt"
    python sim/chunk_measure.py --chunk-mib $chunk --order $order 2>&1 \
      | grep -v -E "^(W11|objc)" > "$f"
    echo "完成 $order chunk=$chunk: $(tail -1 "$f")"
  done
done
echo "全部完成，结果在 $OUT"
