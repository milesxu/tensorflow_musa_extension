import numpy as np
import tensorflow as tf
import tensorflow_musa
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework.graph_util import extract_sub_graph

import test_manual_full_fitting as fitting


MANUAL_NODE = "manual_full_fitting_type_0"
OUTPUT_NODES = ["o_energy", "o_force", "o_virial"]
OUTPUT_TENSORS = [name + ":0" for name in OUTPUT_NODES]


def load_graph_def(with_manual_gradient):
    graph_def = tf.compat.v1.GraphDef()
    with open(fitting.PB, "rb") as handle:
        graph_def.ParseFromString(handle.read())
    if not with_manual_gradient:
        return graph_def

    manual = node_def_pb2.NodeDef()
    manual.name = MANUAL_NODE
    manual.op = "MusaManualGraphFullFitting"
    manual.input.extend(
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
    graph_def.node.extend([manual])

    energy_rewired = 0
    gradient_rewired = 0
    for node in graph_def.node:
        if node.name == "Cast_7":
            for index, input_name in enumerate(node.input):
                if input_name.split(":")[0] == "final_layer_type_0/BiasAdd":
                    node.input[index] = f"{MANUAL_NODE}:0"
                    energy_rewired += 1
        if node.name == "gradients/Slice_3_grad/Pad":
            for index, input_name in enumerate(node.input):
                if (
                    input_name.split(":")[0]
                    == "gradients/Reshape_18_grad/Reshape"
                ):
                    node.input[index] = f"{MANUAL_NODE}:1"
                    gradient_rewired += 1
    if energy_rewired != 1 or gradient_rewired != 1:
        raise RuntimeError(
            "expected one energy and one gradient rewire, got "
            f"{energy_rewired}/{gradient_rewired}"
        )

    pruned = extract_sub_graph(graph_def, OUTPUT_NODES)
    node_names = {node.name for node in pruned.node}
    stale = {
        "layer_0_type_0/MatMul",
        "layer_1_type_0/MatMul",
        "layer_2_type_0/MatMul",
        "final_layer_type_0/MatMul",
        "gradients/Reshape_18_grad/Reshape",
    }
    leaked = sorted(stale & node_names)
    if leaked:
        raise RuntimeError(f"stale type-0 fitting nodes remain: {leaked}")
    if MANUAL_NODE not in node_names:
        raise RuntimeError("manual fitting node was pruned")
    print("manual_pruned_nodes", len(pruned.node))
    print("manual_stale_type0_nodes", 0)

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            tf.import_graph_def(pruned, name="")
    return graph


def build_reference_graph(device):
    graph_def = load_graph_def(False)
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device):
            tf.import_graph_def(graph_def, name="")
    return graph


def fetch_values(graph, feeds, session=None):
    def run(sess):
        tensors = [
            sess.graph.get_tensor_by_name(name) for name in OUTPUT_TENSORS
        ]
        return [np.asarray(value) for value in sess.run(tensors, feed_dict=feeds)]

    if session is not None:
        return run(session)
    with tf.compat.v1.Session(graph=graph) as sess:
        return run(sess)


def print_stats(label, actual, expected):
    return fitting.diff_stats(label, actual, expected)


def check_efv(label, energy, force, virial):
    if energy.shape != (1,) and energy.shape != ():
        print("energy_shape", label, energy.shape)
    if force.shape != (1, 20361):
        raise RuntimeError(f"{label} force shape {force.shape}")
    if virial.shape != (1, 9):
        raise RuntimeError(f"{label} virial shape {virial.shape}")
    if not np.all(np.isfinite(energy)):
        raise RuntimeError(f"{label} energy is not finite")
    if not np.all(np.isfinite(force)):
        raise RuntimeError(f"{label} force is not finite")
    if not np.all(np.isfinite(virial)):
        raise RuntimeError(f"{label} virial is not finite")
    print(
        label,
        "energy",
        energy.tolist(),
        "force_max_abs",
        float(np.max(np.abs(force))),
        "virial_max_abs",
        float(np.max(np.abs(virial))),
    )


def main():
    tf.load_op_library(fitting.CPU_OP_LIB)
    tf.load_op_library(fitting.MUSA_OP_LIB)
    feeds_a = fitting.make_inputs(20261027)
    feeds_b = fitting.make_inputs(20261028)
    feed_map_a = {
        "t_box:0": feeds_a[0],
        "t_coord:0": feeds_a[1],
        "t_type:0": feeds_a[2],
        "t_natoms:0": feeds_a[3],
        "t_mesh:0": feeds_a[4],
    }
    feed_map_b = {
        "t_box:0": feeds_b[0],
        "t_coord:0": feeds_b[1],
        "t_type:0": feeds_b[2],
        "t_natoms:0": feeds_b[3],
        "t_mesh:0": feeds_b[4],
    }

    cpu_graph = build_reference_graph("/CPU:0")
    musa_graph = build_reference_graph("/device:MUSA:0")
    manual_graph = load_graph_def(True)

    with tf.compat.v1.Session(graph=cpu_graph) as sess:
        cpu_a = fetch_values(cpu_graph, feed_map_a, sess)
        cpu_b = fetch_values(cpu_graph, feed_map_b, sess)
    with tf.compat.v1.Session(graph=musa_graph) as sess:
        musa_a = fetch_values(musa_graph, feed_map_a, sess)
        musa_b = fetch_values(musa_graph, feed_map_b, sess)

    with tf.compat.v1.Session(graph=manual_graph) as sess:
        manual_a1 = fetch_values(manual_graph, feed_map_a, sess)
        manual_b = fetch_values(manual_graph, feed_map_b, sess)
        manual_a2 = fetch_values(manual_graph, feed_map_a, sess)
        manual_a3 = fetch_values(manual_graph, feed_map_a, sess)

    for label, values in [
        ("cpu_a", cpu_a),
        ("cpu_b", cpu_b),
        ("musa_a", musa_a),
        ("musa_b", musa_b),
        ("manual_a1", manual_a1),
        ("manual_b", manual_b),
        ("manual_a2", manual_a2),
        ("manual_a3", manual_a3),
    ]:
        check_efv(label, *values)

    for index, name in enumerate(["energy", "force", "virial"]):
        repeat_12 = float(
            np.max(np.abs(manual_a1[index] - manual_a2[index]))
        )
        repeat_23 = float(
            np.max(np.abs(manual_a2[index] - manual_a3[index]))
        )
        input_delta = float(
            np.max(np.abs(manual_a2[index] - manual_b[index]))
        )
        print(f"manual_{name}_repeat_a1_a2_max_abs", repeat_12)
        print(f"manual_{name}_repeat_a2_a3_max_abs", repeat_23)
        print(f"manual_{name}_input_delta_max_abs", input_delta)
    if repeat_12 > 1e-10 or repeat_23 > 1e-10:
        raise RuntimeError(f"manual {name} repeat drift exceeds gate")
    if name == "energy" and input_delta <= 1.0:
        raise RuntimeError("manual energy did not react to input B")
    if name in ("force", "virial") and input_delta <= 1e-3:
        raise RuntimeError(f"manual {name} did not react to input B")
    print("manual_a_repeat_within_tolerance", True)
    print("manual_input_sensitivity", True)

    energy_manual_musa_a = print_stats(
        "energy_manual_a_vs_musa_a", manual_a1[0], musa_a[0]
    )
    energy_manual_musa_b = print_stats(
        "energy_manual_b_vs_musa_b", manual_b[0], musa_b[0]
    )
    energy_manual_cpu_a = print_stats(
        "energy_manual_a_vs_cpu_a", manual_a1[0], cpu_a[0]
    )
    energy_musa_cpu_a = print_stats(
        "energy_musa_a_vs_cpu_a", musa_a[0], cpu_a[0]
    )
    force_manual_musa_a = print_stats(
        "force_manual_a_vs_musa_a", manual_a1[1], musa_a[1]
    )
    force_manual_musa_b = print_stats(
        "force_manual_b_vs_musa_b", manual_b[1], musa_b[1]
    )
    force_manual_cpu_a = print_stats(
        "force_manual_a_vs_cpu_a", manual_a1[1], cpu_a[1]
    )
    force_musa_cpu_a = print_stats(
        "force_musa_a_vs_cpu_a", musa_a[1], cpu_a[1]
    )
    virial_manual_musa_a = print_stats(
        "virial_manual_a_vs_musa_a", manual_a1[2], musa_a[2]
    )
    virial_manual_musa_b = print_stats(
        "virial_manual_b_vs_musa_b", manual_b[2], musa_b[2]
    )
    virial_manual_cpu_a = print_stats(
        "virial_manual_a_vs_cpu_a", manual_a1[2], cpu_a[2]
    )
    virial_musa_cpu_a = print_stats(
        "virial_musa_a_vs_cpu_a", musa_a[2], cpu_a[2]
    )

    for name, metrics in [
        ("energy_a", energy_manual_musa_a),
        ("energy_b", energy_manual_musa_b),
    ]:
        if metrics["max_abs_diff"] > 0.1 or metrics["nrmse"] > 1e-6:
            raise RuntimeError(f"{name} manual/MUSA gate failed")
    for name, metrics in [
        ("force_a", force_manual_musa_a),
        ("force_b", force_manual_musa_b),
    ]:
        if metrics["max_abs_diff"] > 0.5 or metrics["nrmse"] > 2e-3:
            raise RuntimeError(f"{name} manual/MUSA gate failed")
    for name, metrics in [
        ("virial_a", virial_manual_musa_a),
        ("virial_b", virial_manual_musa_b),
    ]:
        if metrics["max_abs_diff"] > 1.0 or metrics["nrmse"] > 5e-5:
            raise RuntimeError(f"{name} manual/MUSA gate failed")

    def relative_gate(name, manual, baseline, max_floor, nrmse_floor):
        if manual["max_abs_diff"] > max(
            baseline["max_abs_diff"] * 1.5, max_floor
        ):
            raise RuntimeError(f"{name} max_abs exceeds MUSA/CPU baseline")
        if manual["nrmse"] > max(
            baseline["nrmse"] * 1.5, nrmse_floor
        ):
            raise RuntimeError(f"{name} NRMSE exceeds MUSA/CPU baseline")

    relative_gate(
        "energy_manual_cpu",
        energy_manual_cpu_a,
        energy_musa_cpu_a,
        5e-3,
        5e-5,
    )
    relative_gate(
        "force_manual_cpu",
        force_manual_cpu_a,
        force_musa_cpu_a,
        0.5,
        5e-3,
    )
    relative_gate(
        "virial_manual_cpu",
        virial_manual_cpu_a,
        virial_musa_cpu_a,
        3.0,
        2e-4,
    )
    print("manual_efv_gates", "PASS")
    print("MANUAL_EFV_REWIRE_PASS")


if __name__ == "__main__":
    main()
