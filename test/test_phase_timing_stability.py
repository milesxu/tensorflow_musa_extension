import os

import numpy as np
import tensorflow as tf
import tensorflow_musa

import test_manual_backward_layout as layout


def main():
    batch = int(os.environ.get("PHASE_TIMING_BATCH", "512"))
    hits = int(os.environ.get("PHASE_TIMING_HITS", "20"))
    arrays = layout.make_arrays(seed=20261028, batch=batch)
    ops = tensorflow_musa.get_musa_ops()

    graph = tf.Graph()
    with graph.as_default():
        placeholders = [
            tf.compat.v1.placeholder(tf.float32, shape=array.shape)
            for array in arrays
        ]
        actual = ops.musa_manual_graph_full_fitting(*placeholders)

    feed = dict(zip(placeholders, arrays))
    with tf.compat.v1.Session(graph=graph) as sess:
        energy = None
        gradient = None
        for _ in range(hits + 1):
            energy, gradient = sess.run(actual, feed_dict=feed)
        if energy.shape != (batch, 1):
            raise RuntimeError(f"unexpected energy shape {energy.shape}")
        if gradient.shape != (1, batch, 1600):
            raise RuntimeError(f"unexpected gradient shape {gradient.shape}")
        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("phase timing output is not finite")
    print(f"PHASE_TIMING_STABILITY_PASS batch={batch} hits={hits}")


if __name__ == "__main__":
    main()
