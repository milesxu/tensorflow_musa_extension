import numpy as np
import tensorflow as tf
import tensorflow_musa


PB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/static-shape-probe/model_static_1536_1rank.pb"
CPU_OP_LIB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/install-deepmd/lib/libdeepmd_op.so"
MUSA_OP_LIB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/install-deepmd/lib/libdeepmd_op_musa.so"
NLOC = 1536
NALL = 6787


def make_inputs():
    rng = np.random.default_rng(20261027)
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
    print(
        f"{label}_max_abs_diff",
        float(np.max(diff)),
    )
    print(
        f"{label}_mean_abs_diff",
        float(np.mean(diff)),
    )
    print(
        f"{label}_p99_abs_diff",
        float(np.percentile(diff, 99)),
    )
    print(f"{label}_rms_diff", rms)
    print(f"{label}_nrmse", rms / max(expected_rms, np.finfo(np.float32).tiny))


def main():
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

    box_value, coord_value, type_value, natoms_value, mesh_value = make_inputs()
    with tf.compat.v1.Session(graph=graph) as sess:
        values = sess.run(
            tensors
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
    arrays = values[: len(tensors)]
    identity_values = values[len(tensors) : -1]
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
    print("input_shapes", [array.shape for array in arrays])
    print("reference_shape", expected.shape, expected.dtype)
    print("tanh_reshape_identity", True)
    print("idt_shapes", arrays[9].shape, arrays[10].shape)
    print(
        "reference_stats",
        {
            "min": float(np.min(expected)),
            "max": float(np.max(expected)),
            "mean": float(np.mean(expected)),
            "rms": float(np.sqrt(np.mean(np.square(expected)))),
        },
    )

    ops = tensorflow_musa.get_musa_ops()
    cap_graph = tf.Graph()
    with cap_graph.as_default():
        with tf.device("/device:MUSA:0"):
            placeholders = [
                tf.compat.v1.placeholder(tf.float32, shape=array.shape)
                for array in arrays
            ]
            actual = ops.musa_manual_graph_full_fitting(*placeholders)
    with tf.compat.v1.Session(graph=cap_graph) as sess:
        result = sess.run(
            actual,
            feed_dict=dict(zip(placeholders, arrays)),
        )

    print("result_shape", result.shape)
    print(
        "result_stats",
        {
            "min": float(np.min(result)),
            "max": float(np.max(result)),
            "mean": float(np.mean(result)),
            "rms": float(np.sqrt(np.mean(np.square(result)))),
        },
    )
    diff_stats("manual_vs_musa", result, expected)
    diff_stats("musa_vs_cpu", expected, cpu_expected)
    diff_stats("manual_vs_cpu", result, cpu_expected)
    if result.shape != expected.shape:
        raise RuntimeError("full fitting output shape mismatch")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("full fitting output is not finite")
    if not np.all(np.isfinite(cpu_expected)):
        raise RuntimeError("NumPy reference output is not finite")
    if not np.allclose(result, expected, rtol=1e-2, atol=1e-2):
        raise RuntimeError("full fitting output mismatch")
    print("MANUAL_FULL_FITTING_PASS")


if __name__ == "__main__":
    main()
