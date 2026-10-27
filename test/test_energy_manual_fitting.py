import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_musa
from tensorflow.core.framework import node_def_pb2


PB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/static-shape-probe/model_static_1536_1rank.pb"
CPU_OP_LIB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/install-deepmd/lib/libdeepmd_op.so"
MUSA_OP_LIB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/install-deepmd/lib/libdeepmd_op_musa.so"
NLOC = 1536
NALL = 6787
MANUAL_NODE = "manual_full_fitting_type_0"


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
    return {
        "t_box:0": box,
        "t_coord:0": coord,
        "t_type:0": np.zeros((1, NALL), dtype=np.int32),
        "t_natoms:0": np.array([NLOC, NALL, NLOC, 0], dtype=np.int32),
        "t_mesh:0": np.zeros((6,), dtype=np.int32),
    }


def load_graph_def(with_manual_fitting):
    graph_def = tf.compat.v1.GraphDef()
    with open(PB, "rb") as handle:
        graph_def.ParseFromString(handle.read())
    if not with_manual_fitting:
        return graph_def

    node = node_def_pb2.NodeDef()
    node.name = MANUAL_NODE
    node.op = "MusaManualGraphFullFitting"
    node.input.extend(
        [
            "Reshape_18",
            "layer_0_type_0/matrix",
            "layer_0_type_0/bias",
            "layer_1_type_0/matrix",
            "layer_1_type_0/bias",
            "layer_2_type_0/matrix",
            "layer_2_type_0/bias",
            "final_layer_type_0/matrix",
            "final_layer_type_0/bias",
            "layer_1_type_0/idt",
            "layer_2_type_0/idt",
        ]
    )
    graph_def.node.extend([node])
    rewired = 0
    for graph_node in graph_def.node:
        if graph_node.name != "Cast_7":
            continue
        for index, input_name in enumerate(graph_node.input):
            if input_name.split(":")[0] == "final_layer_type_0/BiasAdd":
                graph_node.input[index] = MANUAL_NODE
                rewired += 1
    if rewired != 1:
        raise RuntimeError(f"expected one Cast_7 rewrite, got {rewired}")
    return graph_def


def build_graph(graph_def, device):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device):
            tf.import_graph_def(graph_def, name="")
            try:
                manual = graph.get_tensor_by_name(f"{MANUAL_NODE}:0")
            except KeyError:
                manual = None
            return {
                "graph": graph,
                "energy": graph.get_tensor_by_name("o_energy:0"),
                "manual": manual,
            }


def run_session(sess, entry, feeds, fetch_manual=False):
    fetches = [entry["energy"]]
    if fetch_manual:
        fetches.append(entry["manual"])
    values = sess.run(fetches, feed_dict=feeds)
    if fetch_manual:
        return np.asarray(values[0]), np.asarray(values[1])
    return np.asarray(values[0])


def energy_stats(label, actual, expected):
    diff = np.abs(actual - expected)
    max_abs = float(np.max(diff))
    per_atom = float(max_abs / NLOC)
    print(f"{label}_actual", actual.tolist())
    print(f"{label}_expected", expected.tolist())
    print(f"{label}_abs_diff", diff.tolist())
    print(f"{label}_per_atom_abs_diff", per_atom)
    return {"max_abs": max_abs, "per_atom_abs": per_atom}


def main():
    plugin_path = Path(tensorflow_musa.__file__).with_name("libmusa_plugin.so")
    print("tensorflow_version", tf.__version__)
    for path in (plugin_path, PB, CPU_OP_LIB, MUSA_OP_LIB):
        print("sha256", path, file_sha256(path))

    tf.load_op_library(CPU_OP_LIB)
    tf.load_op_library(MUSA_OP_LIB)

    feeds_a = make_inputs(20261027)
    feeds_b = make_inputs(20261028)

    cpu_entry = build_graph(load_graph_def(False), "/CPU:0")
    musa_entry = build_graph(load_graph_def(False), "/device:MUSA:0")
    manual_entry = build_graph(
        load_graph_def(True), "/device:MUSA:0"
    )

    with tf.compat.v1.Session(graph=cpu_entry["graph"]) as cpu_sess:
        energy_cpu_a = run_session(cpu_sess, cpu_entry, feeds_a)
        energy_cpu_b = run_session(cpu_sess, cpu_entry, feeds_b)
    with tf.compat.v1.Session(graph=musa_entry["graph"]) as musa_sess:
        energy_musa_a = run_session(musa_sess, musa_entry, feeds_a)
        energy_musa_b = run_session(musa_sess, musa_entry, feeds_b)
    with tf.compat.v1.Session(graph=manual_entry["graph"]) as manual_sess:
        energy_manual_a1, atom_manual_a1 = run_session(
            manual_sess, manual_entry, feeds_a, fetch_manual=True
        )
        energy_manual_b, atom_manual_b = run_session(
            manual_sess, manual_entry, feeds_b, fetch_manual=True
        )
        energy_manual_a2, atom_manual_a2 = run_session(
            manual_sess, manual_entry, feeds_a, fetch_manual=True
        )

    if not np.array_equal(atom_manual_a1, atom_manual_a2):
        raise RuntimeError("manual atom energy A repeat is not bitwise stable")
    if np.array_equal(atom_manual_a1, atom_manual_b):
        raise RuntimeError("manual atom energy did not react to input B")

    print("manual_atom_energy_shape", atom_manual_a1.shape)
    print("manual_atom_repeat_bitwise", True)
    print("manual_atom_input_sensitivity", True)

    manual_musa_a = energy_stats(
        "manual_a_vs_musa_a", energy_manual_a1, energy_musa_a
    )
    manual_musa_b = energy_stats(
        "manual_b_vs_musa_b", energy_manual_b, energy_musa_b
    )
    manual_cpu_a = energy_stats(
        "manual_a_vs_cpu_a", energy_manual_a1, energy_cpu_a
    )
    manual_cpu_b = energy_stats(
        "manual_b_vs_cpu_b", energy_manual_b, energy_cpu_b
    )
    energy_stats("musa_a_vs_cpu_a", energy_musa_a, energy_cpu_a)
    energy_stats("musa_b_vs_cpu_b", energy_musa_b, energy_cpu_b)
    print(
        "manual_energy_repeat_abs_diff",
        float(np.max(np.abs(energy_manual_a1 - energy_manual_a2))),
    )
    for label, stats in (
        ("manual_musa_a", manual_musa_a),
        ("manual_musa_b", manual_musa_b),
    ):
        if stats["max_abs"] > 0.1 or stats["per_atom_abs"] > 1e-4:
            raise RuntimeError(f"{label} energy difference exceeds gate")
    for label, stats in (
        ("manual_cpu_a", manual_cpu_a),
        ("manual_cpu_b", manual_cpu_b),
    ):
        if stats["per_atom_abs"] > 5e-3:
            raise RuntimeError(f"{label} per-atom energy exceeds gate")
    print("manual_musa_energy_gate", "total<=0.1eV per_atom<=1e-4eV")
    print("manual_cpu_energy_gate", "per_atom<=5e-3eV")
    print("MANUAL_ENERGY_INTEGRATION_PASS")


if __name__ == "__main__":
    main()
