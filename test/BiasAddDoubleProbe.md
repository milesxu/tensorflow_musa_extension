# Double BiasAdd fastpath probe

Branch: `musa/biasadd-double-fastpath`

## Scope

Add a row-major, 2D/NHWC custom kernel for `double` BiasAdd and
compare it with the muDNN broadcast-add path in the same plugin.

The fast path falls back to muDNN for unsupported layouts. The env
`TF_MUSA_DISABLE_DOUBLE_BIASADD_FASTPATH=1` disables the custom path
for A/B.

## Correctness

Passed:

- `(192, 100)` NHWC
- `(3, 5, 7)` NHWC
- `(4, 8, 16, 32)` NHWC
- `(4, 8, 16, 32)` NCHW fallback

All compared against NumPy with exact zero error.

## Timing

TensorFlow session timing, 200 iterations, double:

| shape | custom | muDNN |
|---|---:|---:|
| (192, 100) | 520.6 us | 527.0 us |
| (512, 100) | 539.2 us | 556.8 us |

The custom path is only ~1–3% faster. The full FP64 DeepMD trace did
not show a material improvement.

## Decision

Do not merge into the v19 production path. Keep this branch as a
negative micro-optimization record. The measured bottleneck is not
the BiasAdd arithmetic kernel; host/session/upstream scheduling
dominates this measurement.
