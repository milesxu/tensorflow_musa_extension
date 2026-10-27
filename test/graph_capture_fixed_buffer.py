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
    generation = 20261027
    a0 = np.zeros((4, 4), dtype=np.float32)
    b0 = np.zeros((4, 4), dtype=np.float32)
    a1 = np.arange(16, dtype=np.float32).reshape(4, 4)
    b1 = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
    a2 = np.arange(16, dtype=np.float32).reshape(4, 4) + 100.0
    b2 = np.arange(16, dtype=np.float32).reshape(4, 4) + 0.5
    a3 = np.arange(16, dtype=np.float32).reshape(4, 4) + 200.0
    b3 = np.arange(16, dtype=np.float32).reshape(4, 4) + 2.0

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            a_host = tf.compat.v1.placeholder(tf.float32, shape=(4, 4))
            b_host = tf.compat.v1.placeholder(tf.float32, shape=(4, 4))
            read_template = tf.zeros((4, 4), dtype=tf.float32)
            begin = ops.musa_fixed_graph_capture_begin(
                handle=handle, generation=generation
            )
            with tf.control_dependencies([begin]):
                mm = tf.matmul(a_host, b_host)
                with tf.control_dependencies([mm]):
                    retain = ops.musa_fixed_graph_retain_tensors(
                        a_host,
                        b_host,
                        mm,
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
                    read_template,
                    which=0,
                    handle=handle,
                    generation=generation,
                )
                read_input_b = ops.musa_fixed_graph_read_input(
                    read_template,
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
                    read_template, handle=handle, generation=generation
                )
            with tf.control_dependencies([read_zero]):
                replay = ops.musa_fixed_graph_replay(
                    handle=handle, generation=generation
                )
            with tf.control_dependencies([replay]):
                read_replay = ops.musa_fixed_graph_read_output(
                    read_template, handle=handle, generation=generation
                )
            with tf.control_dependencies([read_replay]):
                release = ops.musa_fixed_graph_release(
                    handle=handle, generation=generation
                )
            with tf.control_dependencies([release]):
                release_done = tf.constant(0, dtype=tf.int32)
            bad_write = ops.musa_fixed_graph_write_input(
                a_host,
                b_host,
                handle=handle,
                generation=generation + 1,
            )
            with tf.control_dependencies([bad_write]):
                bad_write_done = tf.constant(0, dtype=tf.int32)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(capture_done, feed_dict={a_host: a0, b_host: b0})
        print("capture=PASS")

        try:
            sess.run(capture_done, feed_dict={a_host: a0, b_host: b0})
        except tf.errors.OpError as exc:
            print("duplicate_capture_rejected", str(exc).splitlines()[0])
        else:
            raise RuntimeError("duplicate capture was not rejected")

        try:
            sess.run(
                bad_write_done,
                feed_dict={a_host: a1, b_host: b1},
            )
        except tf.errors.OpError as exc:
            print("wrong_generation_rejected", str(exc).splitlines()[0])
        else:
            raise RuntimeError("wrong generation was not rejected")

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
        print("actual_a1", actual_a1.tolist())
        print("actual_b1", actual_b1.tolist())
        if not np.array_equal(actual_a1, a1):
            raise RuntimeError("captured A buffer was not updated")
        if not np.array_equal(actual_b1, b1):
            raise RuntimeError("captured B buffer was not updated")

        print("zero_before_replay_1", zero_before_1.tolist())
        if not np.array_equal(zero_before_1, np.zeros_like(a1)):
            raise RuntimeError("output sentinel #1 was not observed")

        print("diag_1", diag_1.tolist())
        print("result_1", result_1.tolist())
        expected_1 = a1 @ b1
        if not np.array_equal(result_1, expected_1):
            raise RuntimeError("step 1 MatMul mismatch")
        if diag_1.tolist() != [0, 1, 2, 2, 1, 1, 1, 2, generation]:
            raise RuntimeError("unexpected step 1 diagnostics")

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
        print("actual_a2", actual_a2.tolist())
        print("actual_b2", actual_b2.tolist())
        if not np.array_equal(actual_a2, a2):
            raise RuntimeError("captured A buffer was not updated for step 2")
        if not np.array_equal(actual_b2, b2):
            raise RuntimeError("captured B buffer was not updated for step 2")

        print("zero_before_replay_2", zero_before_2.tolist())
        if not np.array_equal(zero_before_2, np.zeros_like(a2)):
            raise RuntimeError("output sentinel #2 was not observed")

        print("diag_2", diag_2.tolist())
        print("result_2", result_2.tolist())
        expected_2 = a2 @ b2
        if not np.array_equal(result_2, expected_2):
            raise RuntimeError("step 2 MatMul mismatch")
        if diag_2.tolist() != [1, 2, 2, 2, 1, 1, 2, 2, generation]:
            raise RuntimeError("unexpected step 2 diagnostics")

        # Recreate the same graph/handle after a successful release. This checks
        # that capture events and graph state are not left in process-global maps.
        sess.run(capture_done, feed_dict={a_host: a0, b_host: b0})
        (
            actual_a3,
            actual_b3,
            zero_before_3,
            diag_3,
            result_3,
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
            feed_dict={a_host: a3, b_host: b3},
        )
        print("actual_a3", actual_a3.tolist())
        print("actual_b3", actual_b3.tolist())
        print("zero_before_replay_3", zero_before_3.tolist())
        print("diag_3", diag_3.tolist())
        print("result_3", result_3.tolist())
        if not np.array_equal(actual_a3, a3):
            raise RuntimeError("captured A buffer was not updated for cycle 2")
        if not np.array_equal(actual_b3, b3):
            raise RuntimeError("captured B buffer was not updated for cycle 2")
        if not np.array_equal(zero_before_3, np.zeros_like(a3)):
            raise RuntimeError("output sentinel #3 was not observed")
        if not np.array_equal(result_3, a3 @ b3):
            raise RuntimeError("cycle 2 MatMul mismatch")
        if diag_3.tolist() != [0, 1, 2, 2, 1, 1, 1, 2, generation]:
            raise RuntimeError("unexpected cycle 2 diagnostics")

    print("GRAPH_CAPTURE_FIXED_BUFFER_PASS")


if __name__ == "__main__":
    main()
