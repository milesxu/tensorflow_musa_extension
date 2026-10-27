import numpy as np
import tensorflow as tf
import tensorflow_musa


def main():
    ops = tensorflow_musa.get_musa_ops()
    for name in (
        "musa_graph_capture_begin",
        "musa_graph_capture_end",
        "musa_graph_replay",
    ):
        if not hasattr(ops, name):
            raise RuntimeError("missing op: %s" % name)

    handle = 17
    a_value = np.arange(16, dtype=np.float32).reshape(4, 4)
    b_value = (np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0)

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            a = tf.constant(a_value)
            b = tf.constant(b_value)
            begin = ops.musa_graph_capture_begin(handle=handle)
            with tf.control_dependencies([begin]):
                mm = tf.matmul(a, b)
            end = ops.musa_graph_capture_end(mm, handle=handle)
            with tf.control_dependencies([end]):
                replay = ops.musa_graph_replay(handle=handle)
            with tf.control_dependencies([replay]):
                out = tf.identity(mm)

    with tf.compat.v1.Session(graph=graph) as sess:
        result = sess.run(out)

    expected = a_value @ b_value
    print("result", result.tolist())
    print("expected", expected.tolist())
    if not np.allclose(result, expected, rtol=0.0, atol=1e-5):
        raise RuntimeError("MatMul graph capture replay mismatch")
    print("GRAPH_CAPTURE_MATMUL_PROBE_PASS")


if __name__ == "__main__":
    main()
