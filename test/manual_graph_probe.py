import numpy as np
import tensorflow as tf
import tensorflow_musa


def main():
    ops = tensorflow_musa.get_musa_ops()
    with tf.Graph().as_default():
        with tf.device("/device:MUSA:0"):
            like = tf.zeros((1,), dtype=tf.float32)
            diagnostics = ops.musa_manual_graph_probe(like)
        with tf.compat.v1.Session() as sess:
            value = sess.run(diagnostics)
    print("diagnostics", np.asarray(value).tolist())
    if np.asarray(value).tolist() != [2, 2, 2, 2, 2]:
        raise RuntimeError("manual graph probe diagnostics mismatch")
    print("MANUAL_GRAPH_PROBE_PASS")


if __name__ == "__main__":
    main()
