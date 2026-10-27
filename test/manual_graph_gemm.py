import numpy as np
import tensorflow as tf
import tensorflow_musa


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

    ops = tensorflow_musa.get_musa_ops()
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            like = tf.zeros((1,), dtype=tf.float32)
            output = ops.musa_manual_graph_gemm(like)
        with tf.compat.v1.Session() as sess:
            result = sess.run(output)

    print("shape", result.shape)
    if not np.allclose(result[0], a1 @ b1, rtol=0.0, atol=1e-12):
        raise RuntimeError("manual GEMM replay #1 mismatch")
    if not np.allclose(result[1], a2 @ b2, rtol=0.0, atol=1e-12):
        raise RuntimeError("manual GEMM replay #2 mismatch")
    print("MANUAL_GEMM_REPLAY_1_OK")
    print("MANUAL_GEMM_REPLAY_2_OK")
    print("MANUAL_GRAPH_GEMM_PASS")


if __name__ == "__main__":
    main()
