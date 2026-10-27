import hashlib
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_musa


PB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/static-shape-probe/model_static_1536_1rank.pb"
CPU_OP_LIB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/install-deepmd/lib/libdeepmd_op.so"
MUSA_OP_LIB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/install-deepmd/lib/libdeepmd_op_musa.so"
NLOC = 1536
NALL = 6787


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_inputs(seed):
    rng = np.random.default_rng(seed)
    box_length = 30.0
    local = rng.uniform(1.0, box_length - 1.0, size=(NLOC, 3))
    ghost_count = NALL - NLOC
    owners = rng.integers(0, NLOC, size=ghost_count)
    directions = rng.normal(size=(ghost_count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.8, 6.0, size=(ghost_count, 1))
    ghosts = local[owners] + directions * radii
    coord = np.concatenate([local, ghosts], axis=0).reshape(1, NALL * 3)
    box = np.array(
        [[box_length, 0.0, 0.0, 0.0, box_length, 0.0, 0.0, 0.0, box_length]],
        dtype=np.float64,
    )
    return (
        box,
        coord,
        np.zeros((1, NALL), dtype=np.int32),
        np.array([NLOC, NALL, NLOC, 0], dtype=np.int32),
        np.zeros((6,), dtype=np.int32),
    )


def numpy_fitting_reference(arrays):
    x, w0, b0, w1, b1, w2, b2, wf, bf, idt1, idt2 = arrays
    h0 = np.tanh(x @ w0 + b0).astype(np.float32)
    h1 = np.tanh(h0 @ w1 + b1).astype(np.float32)
    add2 = (h0 + h1 * idt1).astype(np.float32)
    h2 = np.tanh(add2 @ w2 + b2).astype(np.float32)
    add3 = (add2 + h2 * idt2).astype(np.float32)
    return (add3 @ wf + bf).astype(np.float32)


def diff_stats(label, actual, expected):
    diff = np.abs(actual - expected)
    rms = float(np.sqrt(np.mean(np.square(actual - expected))))
    expected_rms = float(np.sqrt(np.mean(np.square(expected))))
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))
    p99_abs_diff = float(np.percentile(diff, 99))
    nrmse = rms / max(expected_rms, np.finfo(np.float32).tiny)
    print(
        f"{label}_max_abs_diff",
        max_abs_diff,
    )
    print(
        f"{label}_mean_abs_diff",
        mean_abs_diff,
    )
    print(
        f"{label}_p99_abs_diff",
        p99_abs_diff,
    )
    print(f"{label}_rms_diff", rms)
    print(f"{label}_nrmse", nrmse)
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "p99_abs_diff": p99_abs_diff,
        "rms_diff": rms,
        "nrmse": nrmse,
    }


def run_reference(
    sess,
    graph_tensors,
    fitting_tensors,
    identity_pairs,
    reference,
    seed,
):
    box, coord, atom_type, natoms, mesh = graph_tensors
    box_value, coord_value, type_value, natoms_value, mesh_value = (
        make_inputs(seed)
    )
    values = sess.run(
        fitting_tensors
        + [tensor for pair in identity_pairs for tensor in pair]
        + [reference],
        feed_dict={
            box: box_value,
            coord: coord_value,
            atom_type: type_value,
            natoms: natoms_value,
            mesh: mesh_value,
        },
    )
    arrays = values[: len(fitting_tensors)]
    identity_values = values[len(fitting_tensors) : -1]
    expected = values[-1]
    for index in range(3):
        tanh_value = identity_values[index * 2]
        reshape_value = identity_values[index * 2 + 1]
        if not np.array_equal(tanh_value, reshape_value):
            raise RuntimeError(f"layer {index} Tanh/Reshape is not identity")
    if arrays[9].shape != (240,) or arrays[10].shape != (240,):
        raise RuntimeError("residual idt shape mismatch")
    if not np.all(np.isfinite(arrays[9])) or not np.all(
        np.isfinite(arrays[10])
    ):
        raise RuntimeError("residual idt is not finite")
    cpu_expected = numpy_fitting_reference(arrays)
    if not np.all(np.isfinite(cpu_expected)):
        raise RuntimeError("NumPy reference output is not finite")
    return {
        "arrays": arrays,
        "expected": expected,
        "cpu_expected": cpu_expected,
        "seed": seed,
    }


def check_result(label, result, expected, cpu_expected):
    if result.shape != expected.shape:
        raise RuntimeError(f"{label} output shape mismatch")
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} output is not finite")
    manual_vs_musa = diff_stats(
        f"{label}_manual_vs_musa", result, expected
    )
    diff_stats(f"{label}_musa_vs_cpu", expected, cpu_expected)
    manual_vs_cpu = diff_stats(
        f"{label}_manual_vs_cpu", result, cpu_expected
    )
    if (
        manual_vs_cpu["max_abs_diff"] > 2e-5
        or manual_vs_cpu["nrmse"] > 1e-6
    ):
        raise RuntimeError(f"{label} manual fitting differs from NumPy CPU")
    if (
        manual_vs_musa["max_abs_diff"] > 5e-3
        or manual_vs_musa["nrmse"] > 2e-4
    ):
        raise RuntimeError(f"{label} manual fitting differs from MUSA")


def main():
    plugin_path = Path(tensorflow_musa.__file__).with_name("libmusa_plugin.so")
    print("tensorflow_version", tf.__version__)
    for path in (plugin_path, PB, CPU_OP_LIB, MUSA_OP_LIB):
        print("sha256", path, file_sha256(path))

    tf.load_op_library(CPU_OP_LIB)
    tf.load_op_library(MUSA_OP_LIB)
    graph_def = tf.compat.v1.GraphDef()
    with open(PB, "rb") as handle:
        graph_def.ParseFromString(handle.read())

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            tf.import_graph_def(graph_def, name="")
            box = graph.get_tensor_by_name("t_box:0")
            coord = graph.get_tensor_by_name("t_coord:0")
            atom_type = graph.get_tensor_by_name("t_type:0")
            natoms = graph.get_tensor_by_name("t_natoms:0")
            mesh = graph.get_tensor_by_name("t_mesh:0")
            names = [
                "Reshape_18:0",
                "layer_0_type_0/matrix:0",
                "layer_0_type_0/bias:0",
                "layer_1_type_0/matrix:0",
                "layer_1_type_0/bias:0",
                "layer_2_type_0/matrix:0",
                "layer_2_type_0/bias:0",
                "final_layer_type_0/matrix:0",
                "final_layer_type_0/bias:0",
                "layer_1_type_0/idt:0",
                "layer_2_type_0/idt:0",
            ]
            tensors = [graph.get_tensor_by_name(name) for name in names]
            identity_pairs = [
                (
                    graph.get_tensor_by_name(f"layer_{layer}_type_0/Tanh:0"),
                    graph.get_tensor_by_name(f"layer_{layer}_type_0/Reshape:0"),
                )
                for layer in range(3)
            ]
            reference = graph.get_tensor_by_name(
                "final_layer_type_0/BiasAdd:0"
            )

    graph_tensors = [box, coord, atom_type, natoms, mesh]
    with tf.compat.v1.Session(graph=graph) as sess:
        reference_a = run_reference(
            sess,
            graph_tensors,
            tensors,
            identity_pairs,
            reference,
            20261027,
        )
        reference_b = run_reference(
            sess,
            graph_tensors,
            tensors,
            identity_pairs,
            reference,
            20261028,
        )
    arrays_a = reference_a["arrays"]
    arrays_b = reference_b["arrays"]
    expected_a = reference_a["expected"]
    expected_b = reference_b["expected"]
    cpu_expected_a = reference_a["cpu_expected"]
    cpu_expected_b = reference_b["cpu_expected"]
    if np.array_equal(arrays_a[0], arrays_b[0]):
        raise RuntimeError("second fixture did not change fitting input")
    for index in range(1, len(arrays_a)):
        if not np.array_equal(arrays_a[index], arrays_b[index]):
            raise RuntimeError(f"fixture changed weight/idt input {index}")
    print("input_shapes", [array.shape for array in arrays_a])
    print("reference_seeds", reference_a["seed"], reference_b["seed"])
    print("reference_shape", expected_a.shape, expected_a.dtype)
    print("tanh_reshape_identity", True)
    print("idt_shapes", arrays_a[9].shape, arrays_a[10].shape)
    print("fixture_x_changed", True)
    print("fixture_weights_identical", True)
    print(
        "reference_a_stats",
        {
            "min": float(np.min(expected_a)),
            "max": float(np.max(expected_a)),
            "mean": float(np.mean(expected_a)),
            "rms": float(np.sqrt(np.mean(np.square(expected_a)))),
        },
    )
    print(
        "reference_b_stats",
        {
            "min": float(np.min(expected_b)),
            "max": float(np.max(expected_b)),
            "mean": float(np.mean(expected_b)),
            "rms": float(np.sqrt(np.mean(np.square(expected_b)))),
        },
    )

    ops = tensorflow_musa.get_musa_ops()
    cap_graph = tf.Graph()
    with cap_graph.as_default():
        with tf.device("/device:MUSA:0"):
            placeholders = [
                tf.compat.v1.placeholder(tf.float32, shape=array.shape)
                for array in arrays_a
            ]
            actual = ops.musa_manual_graph_full_fitting(*placeholders)
    with tf.compat.v1.Session(graph=cap_graph) as sess:
        start_a1 = time.perf_counter()
        result_a1 = sess.run(
            actual, feed_dict=dict(zip(placeholders, arrays_a))
        )
        start_b = time.perf_counter()
        result_b = sess.run(
            actual, feed_dict=dict(zip(placeholders, arrays_b))
        )
        start_a2 = time.perf_counter()
        result_a2 = sess.run(
            actual, feed_dict=dict(zip(placeholders, arrays_a))
        )
        end_a2 = time.perf_counter()
    if not np.array_equal(result_a1, result_a2):
        raise RuntimeError("cached exec repeat A output changed")
    if np.array_equal(result_a1, result_b):
        raise RuntimeError("cached exec did not react to changed input")

    print("result_shape", result_a1.shape)
    print("cache_calls", 3)
    print("cache_repeat_a_bitwise", True)
    print("cache_input_sensitivity", True)
    cache_miss_ms = (start_b - start_a1) * 1e3
    cache_hit_b_ms = (start_a2 - start_b) * 1e3
    cache_hit_a_ms = (end_a2 - start_a2) * 1e3
    print("cache_miss_ms", cache_miss_ms)
    print("cache_hit_b_ms", cache_hit_b_ms)
    print("cache_hit_a_ms", cache_hit_a_ms)
    print(
        "cache_hit_avg_ms",
        (cache_hit_b_ms + cache_hit_a_ms) / 2.0,
    )
    print(
        "result_a_stats",
        {
            "min": float(np.min(result_a1)),
            "max": float(np.max(result_a1)),
            "mean": float(np.mean(result_a1)),
            "rms": float(np.sqrt(np.mean(np.square(result_a1)))),
        },
    )
    check_result("a", result_a1, expected_a, cpu_expected_a)
    check_result("b", result_b, expected_b, cpu_expected_b)
    print("manual_vs_cpu_gate", "max_abs<=2e-5 nrmse<=1e-6")
    print("manual_vs_musa_gate", "max_abs<=5e-3 nrmse<=2e-4")
    print("MANUAL_FULL_FITTING_CACHE_PASS")


if __name__ == "__main__":
    main()
