import numpy as np
import tensorflow as tf
import tensorflow_musa


def main():
    ops = tensorflow_musa.get_musa_ops()
    handle = 43
    generation = 20261031

    a_abort = np.arange(16, dtype=np.float32).reshape(4, 4)
    b_abort = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
    a_retry = np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0
    b_retry = np.arange(16, dtype=np.float32).reshape(4, 4) + 2.0

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            a_host = tf.compat.v1.placeholder(tf.float32, shape=(4, 4))
            b_host = tf.compat.v1.placeholder(tf.float32, shape=(4, 4))
            out_template = tf.zeros((4, 4), dtype=tf.float32)

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
                    abort_release = ops.musa_fixed_graph_release(
                        handle=handle, generation=generation
                    )
                with tf.control_dependencies([tick]):
                    end_retry = ops.musa_fixed_graph_capture_end(
                        tick, handle=handle, generation=generation
                    )
            with tf.control_dependencies([abort_release]):
                abort_done = tf.constant(0, dtype=tf.int32)
            with tf.control_dependencies([end_retry]):
                capture_retry_done = tf.constant(0, dtype=tf.int32)

            zero = ops.musa_fixed_graph_zero_output(
                handle=handle, generation=generation
            )
            with tf.control_dependencies([zero]):
                read_zero = ops.musa_fixed_graph_read_output(
                    out_template, handle=handle, generation=generation
                )
            replay = ops.musa_fixed_graph_replay(
                handle=handle, generation=generation
            )
            with tf.control_dependencies([replay]):
                read_replay = ops.musa_fixed_graph_read_output(
                    out_template, handle=handle, generation=generation
                )
            with tf.control_dependencies([read_replay]):
                retry_release = ops.musa_fixed_graph_release(
                    handle=handle, generation=generation
                )
            with tf.control_dependencies([retry_release]):
                retry_release_done = tf.constant(0, dtype=tf.int32)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(abort_done, feed_dict={a_host: a_abort, b_host: b_abort})
        print("nonempty_abort=PASS")

        sess.run(
            capture_retry_done,
            feed_dict={a_host: a_retry, b_host: b_retry},
        )
        print("capture_after_abort=PASS")

        zero_before = sess.run(read_zero)
        if not np.array_equal(zero_before, np.zeros((4, 4), dtype=np.float32)):
            raise RuntimeError("retry output sentinel was not observed")

        diag, result = sess.run([replay, read_replay])
        print("diag", diag.tolist())
        if not np.array_equal(result, a_retry @ b_retry):
            raise RuntimeError("retry MatMul result mismatch")
        if diag[0] != 0 or diag[1] != 1 or diag[4] != 1:
            raise RuntimeError("retry diagnostics mismatch")

        sess.run(retry_release_done)

    print("GRAPH_CAPTURE_ABORT_NONEMPTY_PASS")


if __name__ == "__main__":
    main()
