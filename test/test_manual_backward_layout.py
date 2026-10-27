import numpy as np
import tensorflow as tf
import tensorflow_musa

import test_manual_full_fitting as fitting


CPU_OP_LIB = fitting.CPU_OP_LIB
MUSA_OP_LIB = fitting.MUSA_OP_LIB


def make_arrays(seed=1234):
    rng = np.random.default_rng(seed)
    return [
        (rng.normal(size=(1536, 1600)) * 0.01).astype(np.float32),
        (rng.normal(size=(1600, 240)) * 0.02).astype(np.float32),
        (rng.normal(size=240) * 0.01).astype(np.float32),
        (rng.normal(size=(240, 240)) * 0.02).astype(np.float32),
        (rng.normal(size=240) * 0.01).astype(np.float32),
        (rng.normal(size=(240, 240)) * 0.02).astype(np.float32),
        (rng.normal(size=240) * 0.01).astype(np.float32),
        (rng.normal(size=(240, 1)) * 0.02).astype(np.float32),
        np.zeros(1, dtype=np.float32),
        (rng.normal(size=240) * 0.1).astype(np.float32),
        (rng.normal(size=240) * 0.1).astype(np.float32),
    ]


def check(label, actual_energy, actual_grad, cpu_energy, cpu_grad):
    energy_max = float(np.max(np.abs(actual_energy - cpu_energy)))
    grad_max = float(np.max(np.abs(actual_grad - cpu_grad)))
    grad_rms = float(np.sqrt(np.mean((actual_grad - cpu_grad) ** 2)))
    grad_ref_rms = float(np.sqrt(np.mean(cpu_grad**2)))
    grad_nrmse = grad_rms / max(grad_ref_rms, np.finfo(np.float32).tiny)
    print(
        label,
        "energy_max",
        energy_max,
        "grad_max",
        grad_max,
        "grad_rms",
        grad_rms,
        "grad_nrmse",
        grad_nrmse,
    )
    if energy_max > 2e-5:
        raise RuntimeError(f"{label} energy layout mismatch")
    if grad_max > 2e-5 or grad_nrmse > 1e-6:
        raise RuntimeError(f"{label} gradient layout mismatch")


def main():
    tf.load_op_library(CPU_OP_LIB)
    tf.load_op_library(MUSA_OP_LIB)
    ops = tensorflow_musa.get_musa_ops()
    arrays = make_arrays()
    cpu_energy = fitting.numpy_fitting_reference(arrays)
    cpu_grad = fitting.numpy_fitting_gradient(arrays)

    graph = tf.Graph()
    with graph.as_default():
        placeholders = [
            tf.compat.v1.placeholder(tf.float32, shape=array.shape)
            for array in arrays
        ]
        actual = ops.musa_manual_graph_full_fitting(*placeholders)

    with tf.compat.v1.Session(graph=graph) as sess:
        energy_miss, grad_miss = sess.run(
            actual, feed_dict=dict(zip(placeholders, arrays))
        )
        energy_hit, grad_hit = sess.run(
            actual, feed_dict=dict(zip(placeholders, arrays))
        )

    check(
        "miss",
        energy_miss,
        grad_miss,
        cpu_energy,
        cpu_grad,
    )
    check(
        "hit",
        energy_hit,
        grad_hit,
        cpu_energy,
        cpu_grad,
    )
    print("MANUAL_BACKWARD_LAYOUT_PASS")


if __name__ == "__main__":
    main()
