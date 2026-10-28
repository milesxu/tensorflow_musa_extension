#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/mingxu/tmp/tf-musa-stage-graph-probe-20261027/graph-contention-standalone}"
MCC="${MCC:-/usr/local/musa/bin/mcc}"
SRC="$(cd "$(dirname "$0")" && pwd)/graph_contention_repro.mu"
BIN="$ROOT/graph_contention_repro"

mkdir -p "$ROOT"
"$MCC" -O3 -std=c++17 "$SRC" -o "$BIN" \
  -I/usr/local/musa/include \
  -L/usr/local/musa/lib -L/usr/local/musa/lib64 \
  -Wl,-rpath,/usr/local/musa/lib:/usr/local/musa/lib64 \
  -lmusart -lmublas -lpthread

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
export MTHREADS_VISIBLE_DEVICES="${MTHREADS_VISIBLE_DEVICES:-0}"

for mode in graph direct; do
  for noise_blocks in 0 64 256 512 1024; do
    "$BIN" --mode "$mode" --noise-blocks "$noise_blocks" \
      --noise-iterations 50000 --samples 50 --warmup 10 \
      >"$ROOT/${mode}_noise${noise_blocks}.log" 2>&1
    grep '^SUMMARY' "$ROOT/${mode}_noise${noise_blocks}.log"
  done
done
