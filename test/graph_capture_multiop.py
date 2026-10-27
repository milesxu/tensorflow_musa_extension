import numpy as np
import tensorflow as tf
import tensorflow_musa


def _reference(a_value, b_value):
    mm = a_value @ b_value
    return np.pad(
        np.concatenate([np.tanh(mm), mm], axis=0),
        ((0, 0), (1, 1)),
    )


def main():
    ops = tensorflow_musa.get_musa_ops()
    handle = 37
    generation = 20261029

    a0 = np.zeros((4, 4), dtype=np.float64)
    b0 = np.zeros((4, 4), dtype=np.float64)
    a1 = (np.arange(16, dtype=np.float64).reshape(4, 4) - 7.5) / 8.0
    b1 = (np.arange(16, dtype=np.float64).reshape(4, 4) + 1.0) / 9.0
    a2 = (np.arange(16, dtype=np.float64).reshape(4, 4) + 20.0) / 11.0
    b2 = (np.arange(16, dtype=np.float64).reshape(4, 4) + 0.25) / 7.0

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            a_host = tf.compat.v1.placeholder(tf.float64, shape=(4, 4))
            b_host = tf.compat.v1.placeholder(tf.float64, shape=(4, 4))
            ab_template = tf.zeros((4, 4), dtype=tf.float64)
            out_template = tf.zeros((8, 6), dtype=tf.float64)

            begin = ops.musa_fixed_graph_capture_begin(
                handle=handle, generation=generation
            )
            with tf.control_dependencies([begin]):
                mm = tf.matmul(a_host, b_host)
                tanh = tf.tanh(mm)
                concat = tf.concat([tanh, mm], axis=0)
                padded = tf.pad(concat, [[0, 0], [1, 1]])
                with tf.control_dependencies([padded]):
                    retain = ops.musa_fixed_graph_retain_tensors(
                        a_host,
                        b_host,
                        padded,
                        handle=handle,
                        generation=generation,
                    )
                with tf.control_dependencies([retain]):
                    tick = ops.musa_fixed_graph_counter_tick(
                        handle=handle, generation=generation
                    )
                with tf.control_dependencies([tick]):
                    end = ops.musa_fixed_graph_capture_end(
                        tick, handle=handle, generation=generation
                    )
            with tf.control_dependencies([end]):
                capture_done = tf.constant(0, dtype=tf.int32)

            write_input = ops.musa_fixed_graph_write_input(
                a_host,
                b_host,
                handle=handle,
                generation=generation,
            )
            with tf.control_dependencies([write_input]):
                read_input_a = ops.musa_fixed_graph_read_input(
                    ab_template,
                    which=0,
                    handle=handle,
                    generation=generation,
                )
                read_input_b = ops.musa_fixed_graph_read_input(
                    ab_template,
                    which=1,
                    handle=handle,
                    generation=generation,
                )
            with tf.control_dependencies([read_input_a, read_input_b]):
                zero = ops.musa_fixed_graph_zero_output(
                    handle=handle, generation=generation
                )
            with tf.control_dependencies([zero]):
                read_zero = ops.musa_fixed_graph_read_output(
                    out_template, handle=handle, generation=generation
                )
            with tf.control_dependencies([read_zero]):
                replay = ops.musa_fixed_graph_replay(
                    handle=handle, generation=generation
                )
            with tf.control_dependencies([replay]):
                read_replay = ops.musa_fixed_graph_read_output(
                    out_template, handle=handle, generation=generation
                )
            with tf.control_dependencies([read_replay]):
                release = ops.musa_fixed_graph_release(
                    handle=handle, generation=generation
                )
            with tf.control_dependencies([release]):
                release_done = tf.constant(0, dtype=tf.int32)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(capture_done, feed_dict={a_host: a0, b_host: b0})
        print("multi_op_capture=PASS")

        (
            actual_a1,
            actual_b1,
            zero_before_1,
            diag_1,
            result_1,
        ) = sess.run(
            [read_input_a, read_input_b, read_zero, replay, read_replay],
            feed_dict={a_host: a1, b_host: b1},
        )
        print("diag_1", diag_1.tolist())
        if not np.array_equal(actual_a1, a1):
            raise RuntimeError("captured A buffer was not updated")
        if not np.array_equal(actual_b1, b1):
            raise RuntimeError("captured B buffer was not updated")
        if not np.array_equal(zero_before_1, np.zeros((8, 6), dtype=np.float64)):
            raise RuntimeError("output sentinel #1 was not observed")
        if not np.allclose(result_1, _reference(a1, b1), rtol=1e-12, atol=1e-12):
            raise RuntimeError("step 1 multi-op result mismatch")

        (
            actual_a2,
            actual_b2,
            zero_before_2,
            diag_2,
            result_2,
            _,
        ) = sess.run(
            [
                read_input_a,
                read_input_b,
                read_zero,
                replay,
                read_replay,
                release_done,
            ],
            feed_dict={a_host: a2, b_host: b2},
        )
        print("diag_2", diag_2.tolist())
        if not np.array_equal(actual_a2, a2):
            raise RuntimeError("captured A buffer was not updated for step 2")
        if not np.array_equal(actual_b2, b2):
            raise RuntimeError("captured B buffer was not updated for step 2")
        if not np.array_equal(zero_before_2, np.zeros((8, 6), dtype=np.float64)):
            raise RuntimeError("output sentinel #2 was not observed")
        if not np.allclose(result_2, _reference(a2, b2), rtol=1e-12, atol=1e-12):
            raise RuntimeError("step 2 multi-op result mismatch")

        d1 = diag_1.tolist()
        d2 = diag_2.tolist()
        if d1[0] != 0 or d1[1] != 1 or d1[4] != 1:
            raise RuntimeError("unexpected step 1 counters")
        if d2[0] != 1 or d2[1] != 2 or d2[4] != 1:
            raise RuntimeError("unexpected step 2 counters")
        if d1[3] < 4 or d2[3] < 4:
            raise RuntimeError("multi-op graph did not contain expected kernels")
        if d1[7] < 1 or d2[7] < 1:
            raise RuntimeError("capture events were not retained")
        if d1[8] != generation or d2[8] != generation:
            raise RuntimeError("generation mismatch")

    print("GRAPH_CAPTURE_MULTIOP_PASS")


if __name__ == "__main__":
    main()
