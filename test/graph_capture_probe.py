import numpy as np
import tensorflow as tf
import tensorflow_musa


def main():
    ops = tensorflow_musa.get_musa_ops()
    if not hasattr(ops, "musa_graph_capture_probe"):
        raise RuntimeError(
            "MusaGraphCaptureProbe missing; ops=%s"
            % sorted(name for name in dir(ops) if "musa" in name.lower())[:20]
        )

    x = tf.constant(np.arange(16, dtype=np.float32))
    with tf.device("/device:MUSA:0"):
        y = ops.musa_graph_capture_probe(x)
    result = y.numpy()
    # Probe initializes y=0, captures one accumulate kernel, and replays twice.
    # A capture-time execution would incorrectly add a third increment.
    expected = (np.arange(16, dtype=np.float32) + 1.0) * 2.0
    print("result", result.tolist())
    print("expected", expected.tolist())
    if not np.array_equal(result, expected):
        raise RuntimeError("graph capture replay mismatch")
    print("GRAPH_CAPTURE_PROBE_PASS")


if __name__ == "__main__":
    main()
