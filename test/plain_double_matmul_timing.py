import time

import numpy as np
import tensorflow as tf
import tensorflow_musa  # noqa: F401


def main():
    a_value = (np.arange(16, dtype=np.float64).reshape(4, 4) - 7.5) / 8.0
    b_value = (np.arange(16, dtype=np.float64).reshape(4, 4) + 1.0) / 9.0

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            a = tf.compat.v1.placeholder(tf.float64, shape=(4, 4))
            b = tf.compat.v1.placeholder(tf.float64, shape=(4, 4))
            out = tf.matmul(a, b)

    started = time.perf_counter()
    with tf.compat.v1.Session(graph=graph) as sess:
        first_started = time.perf_counter()
        result = sess.run(out, feed_dict={a: a_value, b: b_value})
        first_done = time.perf_counter()
        result_again = sess.run(out, feed_dict={a: a_value, b: b_value})
        second_done = time.perf_counter()

    expected = a_value @ b_value
    print("session_setup_s", first_started - started)
    print("first_run_s", first_done - first_started)
    print("second_run_s", second_done - first_done)
    if not np.allclose(result, expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("first FP64 MatMul mismatch")
    if not np.allclose(result_again, expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("second FP64 MatMul mismatch")
    print("PLAIN_DOUBLE_MATMUL_TIMING_PASS")


if __name__ == "__main__":
    main()
