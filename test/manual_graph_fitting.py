import numpy as np
import tensorflow as tf
import tensorflow_musa


def _reference(a_value, b_value):
    value = a_value @ b_value
    return np.tanh(value), np.sum(np.tanh(value))


def main():
    a1 = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=np.float64,
    )
    b1 = np.array(
        [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]],
        dtype=np.float64,
    )
    a2 = np.array(
        [[2, 3, 5, 7], [11, 13, 17, 19], [23, 29, 31, 37], [41, 43, 47, 53]],
        dtype=np.float64,
    )
    b2 = np.array(
        [[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 8], [9, 7, 9, 3]],
        dtype=np.float64,
    )

    expected_tanh_1, expected_sum_1 = _reference(a1, b1)
    expected_tanh_2, expected_sum_2 = _reference(a2, b2)

    ops = tensorflow_musa.get_musa_ops()
    with tf.Graph().as_default():
        with tf.device("/device:MUSA:0"):
            like = tf.zeros((1,), dtype=tf.float32)
            output = ops.musa_manual_graph_fitting(like)
        with tf.compat.v1.Session() as sess:
            result = sess.run(output)

    print("shape", result.shape)
    if not np.allclose(
        result[0, :16], expected_tanh_1.reshape(-1), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("manual fitting tanh #1 mismatch")
    if not np.allclose(result[0, 16], expected_sum_1, rtol=0.0, atol=1e-12):
        raise RuntimeError("manual fitting sum #1 mismatch")
    if not np.allclose(
        result[1, :16], expected_tanh_2.reshape(-1), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("manual fitting tanh #2 mismatch")
    if not np.allclose(result[1, 16], expected_sum_2, rtol=0.0, atol=1e-12):
        raise RuntimeError("manual fitting sum #2 mismatch")
    print("MANUAL_FITTING_REPLAY_1_OK")
    print("MANUAL_FITTING_REPLAY_2_OK")
    print("MANUAL_GRAPH_FITTING_PASS")


if __name__ == "__main__":
    main()
