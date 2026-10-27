import numpy as np
import tensorflow as tf
import tensorflow_musa


def main():
    ops = tensorflow_musa.get_musa_ops()
    required = (
        "musa_fixed_graph_capture_begin",
        "musa_fixed_graph_retain_tensors",
        "musa_fixed_graph_counter_tick",
        "musa_fixed_graph_capture_end",
        "musa_fixed_graph_replay",
        "musa_fixed_graph_zero_output",
        "musa_fixed_graph_write_input",
        "musa_fixed_graph_read_output",
        "musa_fixed_graph_read_input",
        "musa_fixed_graph_release",
    )
    for name in required:
        if not hasattr(ops, name):
            raise RuntimeError("missing op: %s" % name)

    handle = 23
    a0 = np.zeros((4, 4), dtype=np.float32)
    b0 = np.zeros((4, 4), dtype=np.float32)
    a1 = np.arange(16, dtype=np.float32).reshape(4, 4)
    b1 = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
    a2 = np.arange(16, dtype=np.float32).reshape(4, 4) + 100.0
    b2 = np.arange(16, dtype=np.float32).reshape(4, 4) + 0.5

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            a_host = tf.compat.v1.placeholder(tf.float32, shape=(4, 4))
            b_host = tf.compat.v1.placeholder(tf.float32, shape=(4, 4))
            read_template = tf.zeros((4, 4), dtype=tf.float32)
            begin = ops.musa_fixed_graph_capture_begin(handle=handle)
            with tf.control_dependencies([begin]):
                mm = tf.matmul(a_host, b_host)
                with tf.control_dependencies([mm]):
                    retain = ops.musa_fixed_graph_retain_tensors(
                        a_host, b_host, mm, handle=handle
                    )
                with tf.control_dependencies([retain]):
                    tick = ops.musa_fixed_graph_counter_tick(handle=handle)
                with tf.control_dependencies([tick]):
                    end = ops.musa_fixed_graph_capture_end(
                        tick, handle=handle
                    )

            replay = ops.musa_fixed_graph_replay(handle=handle)
            zero = ops.musa_fixed_graph_zero_output(handle=handle)
            write_input = ops.musa_fixed_graph_write_input(
                a_host, b_host, handle=handle
            )
            read_zero = ops.musa_fixed_graph_read_output(
                read_template, handle=handle
            )
            read_replay = ops.musa_fixed_graph_read_output(
                read_template, handle=handle
            )
            read_input_a = ops.musa_fixed_graph_read_input(
                read_template, which=0, handle=handle
            )
            read_input_b = ops.musa_fixed_graph_read_input(
                read_template, which=1, handle=handle
            )
            with tf.control_dependencies([zero]):
                zero_then_read = tf.identity(read_zero)
            with tf.control_dependencies([replay]):
                replay_then_read = tf.identity(read_replay)
            release = ops.musa_fixed_graph_release(handle=handle)
            release_done = tf.identity(release)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(end, feed_dict={a_host: a0, b_host: b0})

        sess.run(write_input, feed_dict={a_host: a1, b_host: b1})
        actual_a1, actual_b1 = sess.run([read_input_a, read_input_b])
        print("actual_a1", actual_a1.tolist())
        print("actual_b1", actual_b1.tolist())
        if not np.array_equal(actual_a1, a1):
            raise RuntimeError("captured A buffer was not updated")
        if not np.array_equal(actual_b1, b1):
            raise RuntimeError("captured B buffer was not updated")

        zero_before_1 = sess.run(zero_then_read)
        print("zero_before_replay_1", zero_before_1.tolist())
        if not np.array_equal(zero_before_1, np.zeros_like(a1)):
            raise RuntimeError("output sentinel #1 was not observed")

        diag_1, result_1 = sess.run([replay, replay_then_read])
        print("diag_1", diag_1.tolist())
        print("result_1", result_1.tolist())
        expected_1 = a1 @ b1
        if not np.array_equal(result_1, expected_1):
            raise RuntimeError("step 1 MatMul mismatch")
        if diag_1.tolist() != [0, 1, 2, 2, 1, 1, 1]:
            raise RuntimeError("unexpected step 1 diagnostics")

        sess.run(write_input, feed_dict={a_host: a2, b_host: b2})

        zero_before_2 = sess.run(zero_then_read)
        print("zero_before_replay_2", zero_before_2.tolist())
        if not np.array_equal(zero_before_2, np.zeros_like(a2)):
            raise RuntimeError("output sentinel #2 was not observed")

        diag_2, result_2 = sess.run([replay, replay_then_read])
        print("diag_2", diag_2.tolist())
        print("result_2", result_2.tolist())
        expected_2 = a2 @ b2
        if not np.array_equal(result_2, expected_2):
            raise RuntimeError("step 2 MatMul mismatch")
        if diag_2.tolist() != [1, 2, 2, 2, 1, 1, 2]:
            raise RuntimeError("unexpected step 2 diagnostics")

        sess.run(release_done)

    print("GRAPH_CAPTURE_FIXED_BUFFER_PASS")


if __name__ == "__main__":
    main()
