import hashlib
import os

import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework.graph_util import extract_sub_graph


SOURCE_PB = (
    "/data/mingxu/results/ningde_v8_20261021/"
    "tf-musa-mpi-v8-20261027/static-shape-probe/"
    "model_static_1536_1rank.pb"
)
OUTPUT_PB = os.environ.get(
    "MUSA_REWIRED_PB",
    "/data/mingxu/tmp/tf-musa-stage-graph-probe-20261027/"
    "model_static_1536_1rank_manual_efv.pb",
)
MANUAL_NODE = "manual_full_fitting_type_0"


def make_rewired_graph_def(prune=True):
    graph_def = tf.compat.v1.GraphDef()
    with open(SOURCE_PB, "rb") as handle:
        graph_def.ParseFromString(handle.read())

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
            f"unexpected rewire count {energy_rewired}/{gradient_rewired}"
        )

    rewired = extract_sub_graph(
        graph_def, ["o_energy", "o_force", "o_virial"]
    ) if prune else graph_def
    names = {node.name for node in rewired.node}
    stale = {
        "layer_0_type_0/MatMul",
        "layer_0_type_0/BiasAdd",
        "layer_0_type_0/Tanh",
        "layer_0_type_0/Reshape",
        "layer_1_type_0/MatMul",
        "layer_1_type_0/BiasAdd",
        "layer_1_type_0/Tanh",
        "layer_1_type_0/Reshape",
        "layer_1_type_0/mul",
        "layer_2_type_0/MatMul",
        "layer_2_type_0/BiasAdd",
        "layer_2_type_0/Tanh",
        "layer_2_type_0/Reshape",
        "layer_2_type_0/mul",
        "add_2",
        "add_3",
        "final_layer_type_0/MatMul",
        "final_layer_type_0/BiasAdd",
        "gradients/Reshape_18_grad/Reshape",
    }
    leaked = sorted(stale & names)
    if prune and leaked:
        raise RuntimeError(f"stale type-0 nodes remain: {leaked}")
    if MANUAL_NODE not in names:
        raise RuntimeError("manual node was pruned")
    return rewired


def main():
    prune = os.environ.get("MUSA_REWIRE_PRUNE", "1") != "0"
    graph_def = make_rewired_graph_def(prune=prune)
    payload = graph_def.SerializeToString()
    os.makedirs(os.path.dirname(OUTPUT_PB), exist_ok=True)
    with open(OUTPUT_PB, "wb") as handle:
        handle.write(payload)
    print("rewired_pb", OUTPUT_PB)
    print("rewired_nodes", len(graph_def.node))
    print("rewired_pruned", prune)
    print("rewired_bytes", len(payload))
    print("rewired_sha256", hashlib.sha256(payload).hexdigest())
    print("REWIRED_MODEL_PASS")


if __name__ == "__main__":
    main()
