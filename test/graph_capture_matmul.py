import numpy as np
import tensorflow as tf
import tensorflow_musa


def main():
    ops = tensorflow_musa.get_musa_ops()
    for name in (
        "musa_graph_capture_begin",
        "musa_graph_counter_tick",
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
            # Keep MatMul out of Grappler constant folding so the captured graph
            # must contain the real TF MuBLAS kernel.
            a = tf.compat.v1.placeholder(tf.float32, shape=a_value.shape)
            b = tf.compat.v1.placeholder(tf.float32, shape=b_value.shape)
            begin = ops.musa_graph_capture_begin(handle=handle)
            with tf.control_dependencies([begin]):
                mm = tf.matmul(a, b)
                with tf.control_dependencies([mm]):
                    tick = ops.musa_graph_counter_tick(handle=handle)
                with tf.control_dependencies([tick]):
                    end = ops.musa_graph_capture_end(mm, handle=handle)
            with tf.control_dependencies([end]):
                replay = ops.musa_graph_replay(handle=handle)
            with tf.control_dependencies([replay]):
                out = tf.identity(mm)

    with tf.compat.v1.Session(graph=graph) as sess:
        result, diagnostics = sess.run(
            [out, replay], feed_dict={a: a_value, b: b_value}
        )

    expected = a_value @ b_value
    print("result", result.tolist())
    print("expected", expected.tolist())
    print(
        "diagnostics",
        {
            "counter_before_replay": int(diagnostics[0]),
            "counter_after_replay": int(diagnostics[1]),
            "graph_nodes": int(diagnostics[2]),
            "kernel_nodes": int(diagnostics[3]),
        },
    )
    if not np.allclose(result, expected, rtol=0.0, atol=1e-5):
        raise RuntimeError("MatMul graph capture replay mismatch")
    if int(diagnostics[0]) != 0:
        raise RuntimeError("counter kernel executed during capture")
    if int(diagnostics[1]) != 1:
        raise RuntimeError("counter kernel did not execute exactly once")
    if int(diagnostics[2]) < 2 or int(diagnostics[3]) < 2:
        raise RuntimeError(
            "captured graph lacks expected MatMul and counter kernel nodes"
        )
    print("GRAPH_CAPTURE_MATMUL_PROBE_PASS")


if __name__ == "__main__":
    main()
