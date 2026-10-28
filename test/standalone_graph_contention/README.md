# Standalone MUSA graph contention reproducer

This case isolates the graph replay behavior observed in the DeepMD
manual fitting probe without TensorFlow, DeepMD, or LAMMPS.

The captured graph contains 26 nodes: 11 muBLAS SGEMM nodes and 15
MUSA kernel nodes. It mirrors the composition of the manual fitting
graph. `--mode graph` replays the captured graph; `--mode direct`
launches the same 26 operations directly.

The background `--noise-blocks` workload runs a separate stream with
the same stream lifetime as the measured launches. Noise level 0 is
the isolated control.

Build/run:

```bash
bash standalone_graph_contention_run.sh \
  /data/mingxu/tmp/tf-musa-stage-graph-probe-20261027/graph-contention-standalone
```

Optional stream-priority probe:

```bash
./graph_contention_repro --mode graph --noise-blocks 512 \
  --noise-iterations 50000 --samples 50 --warmup 10 \
  --victim-priority -1
```

The valid priority range is printed in `GRAPH_CONTENTION_REPRO`.

On the tested S5000 stack, `--victim-priority -2` did not remove the
long-tail spikes: with `noise_blocks=512`, the high-priority graph
still had `4/50` samples over `2 ms` (p95 `15.97 ms`, max `17.09 ms`).
This suggests priority does not preempt a background kernel that is
already executing; report this as an observation, not yet a confirmed
runtime mechanism.

Expected output:

```text
SUMMARY mode=graph noise_blocks=0 ...
SUMMARY mode=direct noise_blocks=0 ...
```

Report `nodes`, `kernels`, `device_median_us`, `device_p95_us`,
`device_p99_us`, `device_max_us`, `device_spikes_over_2ms`, and
`launch_median_us` for each mode/noise pair.
