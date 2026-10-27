import time

import numpy as np
import tensorflow as tf
import tensorflow_musa  # noqa: F401


def main():
    value = np.arange(16, dtype=np.float64).reshape(4, 4)
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            x = tf.constant(value)
            out = tf.identity(x)

    with tf.compat.v1.Session(graph=graph) as sess:
        first_started = time.perf_counter()
        result = sess.run(out)
        first_done = time.perf_counter()
        result_again = sess.run(out)
        second_done = time.perf_counter()

    print("first_run_s", first_done - first_started)
    print("second_run_s", second_done - first_done)
    if not np.array_equal(result, value):
        raise RuntimeError("first constant identity mismatch")
    if not np.array_equal(result_again, value):
        raise RuntimeError("second constant identity mismatch")
    print("PLAIN_CONSTANT_IDENTITY_TIMING_PASS")


if __name__ == "__main__":
    main()
