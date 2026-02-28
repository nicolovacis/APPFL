#!/usr/bin/env bash
# Run compare_four_approaches for MNIST, CIFAR10, CIFAR100
# 5 clients, 10 local epochs, 20 rounds
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs_sass"
mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR/examples"

# Fix libstdc++ if needed (Docker env)
LIBSTDCXX="$HOME/miniforge3/lib/libstdc++.so.6"
[ -f "$LIBSTDCXX" ] && export LD_PRELOAD="$LIBSTDCXX"

LOG="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

nohup python -u compare_four_approaches.py \
    --datasets MNIST CIFAR10 CIFAR100 \
    --num_clients 5 \
    --local_epochs 10 \
    --rounds 20 \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "Log: $LOG"
echo "tail -f $LOG"
