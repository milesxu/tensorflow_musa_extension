import tensorflow as tf
import tensorflow_musa


def main():
    ops = tensorflow_musa.get_musa_ops()
    handle = 31
    generation = 20261028

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            begin = ops.musa_fixed_graph_capture_begin(
                handle=handle, generation=generation
            )
            with tf.control_dependencies([begin]):
                release = ops.musa_fixed_graph_release(
                    handle=handle, generation=generation
                )
            with tf.control_dependencies([release]):
                done = tf.constant(0, dtype=tf.int32)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(done)
        print("release_while_capturing_cycle_1=PASS")
        # The same handle/generation must be reusable after abort/cleanup.
        sess.run(done)
        print("release_while_capturing_cycle_2=PASS")

    print("GRAPH_CAPTURE_RELEASE_WHILE_CAPTURING_PASS")


if __name__ == "__main__":
    main()
