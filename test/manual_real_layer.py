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
    atom_type = np.zeros((1, NALL), dtype=np.int32)
    natoms = np.array([NLOC, NALL, NLOC, 0], dtype=np.int32)
    mesh = np.zeros((6,), dtype=np.int32)
    return box, coord, atom_type, natoms, mesh


def main():
    tf.load_op_library(CPU_OP_LIB)
    tf.load_op_library(MUSA_OP_LIB)
    graph_def = tf.compat.v1.GraphDef()
    with open(PB, "rb") as handle:
        graph_def.ParseFromString(handle.read())

    ref_graph = tf.Graph()
    with ref_graph.as_default():
        with tf.device("/device:MUSA:0"):
            tf.import_graph_def(graph_def, name="")
            ref_box = ref_graph.get_tensor_by_name("t_box:0")
            ref_coord = ref_graph.get_tensor_by_name("t_coord:0")
            ref_type = ref_graph.get_tensor_by_name("t_type:0")
            ref_natoms = ref_graph.get_tensor_by_name("t_natoms:0")
            ref_mesh = ref_graph.get_tensor_by_name("t_mesh:0")
            layer_input = ref_graph.get_tensor_by_name("Reshape_18:0")
            layer_matrix = ref_graph.get_tensor_by_name(
                "layer_0_type_0/matrix:0"
            )
            layer_bias = ref_graph.get_tensor_by_name("layer_0_type_0/bias:0")
            layer_pre = ref_graph.get_tensor_by_name(
                "layer_0_type_0/BiasAdd:0"
            )
            layer_ref = ref_graph.get_tensor_by_name(
                "layer_0_type_0/Tanh:0"
            )

    box, coord, atom_type, natoms, mesh = make_inputs()
    with tf.compat.v1.Session(graph=ref_graph) as sess:
        x, weight, bias, expected_pre, expected = sess.run(
            [layer_input, layer_matrix, layer_bias, layer_pre, layer_ref],
            feed_dict={
                ref_box: box,
                ref_coord: coord,
                ref_type: atom_type,
                ref_natoms: natoms,
                ref_mesh: mesh,
            },
        )
    print("layer_input", x.shape, x.dtype)
    print("layer_matrix", weight.shape, weight.dtype)
    print("layer_bias", bias.shape, bias.dtype)
    print("layer_pre_reference", expected_pre.shape, expected_pre.dtype)
    print("layer_reference", expected.shape, expected.dtype)

    ops = tensorflow_musa.get_musa_ops()
    cap_graph = tf.Graph()
    with cap_graph.as_default():
        with tf.device("/device:MUSA:0"):
            input_ph = tf.compat.v1.placeholder(tf.float32, shape=x.shape)
            weight_ph = tf.compat.v1.placeholder(tf.float32, shape=weight.shape)
            bias_ph = tf.compat.v1.placeholder(tf.float32, shape=bias.shape)
            actual = ops.musa_manual_graph_real_layer(
                input_ph, weight_ph, bias_ph
            )
    with tf.compat.v1.Session(graph=cap_graph) as sess:
        result = sess.run(
            actual,
            feed_dict={
                input_ph: x,
                weight_ph: weight,
                bias_ph: bias,
            },
        )

    print("result_shape", result.shape)
    if result.shape != (4,) + expected.shape:
        raise RuntimeError("real layer output shape mismatch")
    if not np.array_equal(result[0], result[2]):
        raise RuntimeError("real layer pre-activation replay is not deterministic")
    if not np.array_equal(result[1], result[3]):
        raise RuntimeError("real layer tanh replay is not deterministic")
    pre_diff = np.abs(result[0] - expected_pre)
    tanh_diff = np.abs(result[1] - expected)
    print("pre_max_abs_diff", float(np.max(pre_diff)))
    print("pre_mean_abs_diff", float(np.mean(pre_diff)))
    print("pre_p99_abs_diff", float(np.percentile(pre_diff, 99)))
    print("tanh_max_abs_diff", float(np.max(tanh_diff)))
    print("tanh_mean_abs_diff", float(np.mean(tanh_diff)))
    print("tanh_p99_abs_diff", float(np.percentile(tanh_diff, 99)))
    if not np.allclose(result[0], expected_pre, rtol=1e-2, atol=1e-2):
        raise RuntimeError("real layer pre-activation mismatch")
    if not np.allclose(result[1], expected, rtol=1e-2, atol=1e-2):
        raise RuntimeError("real layer output mismatch")
    print("MANUAL_REAL_LAYER_PASS")


if __name__ == "__main__":
    main()
