import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_musa

import test_manual_full_fitting as fitting


DUMP_DIR = os.environ.get(
    "MUSA_BACKWARD_DUMP_DIR",
    "/data/mingxu/tmp/tf-musa-stage-graph-probe-20261027/backward_intermediates",
)


def build_reference_graph():
    graph_def = tf.compat.v1.GraphDef()
    with open(fitting.PB, "rb") as handle:
        graph_def.ParseFromString(handle.read())
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            tf.import_graph_def(graph_def, name="")
            graph_tensors = [
                graph.get_tensor_by_name(name)
                for name in [
                    "t_box:0",
                    "t_coord:0",
                    "t_type:0",
                    "t_natoms:0",
                    "t_mesh:0",
                ]
            ]
            fitting_tensors = [
                graph.get_tensor_by_name(name + ":0")
                for name in [
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
            ]
            identity_pairs = [
                (
                    graph.get_tensor_by_name(
                        f"layer_{layer}_type_0/Tanh:0"
                    ),
                    graph.get_tensor_by_name(
                        f"layer_{layer}_type_0/Reshape:0"
                    ),
                )
                for layer in range(3)
            ]
            reference = graph.get_tensor_by_name(
                "final_layer_type_0/BiasAdd:0"
            )
            grad_reference = graph.get_tensor_by_name(
                "gradients/Reshape_18_grad/Reshape:0"
            )
    return {
        "graph": graph,
        "graph_tensors": graph_tensors,
        "fitting_tensors": fitting_tensors,
        "identity_pairs": identity_pairs,
        "reference": reference,
        "grad_reference": grad_reference,
    }


def read_dump(name, shape):
    path = Path(DUMP_DIR) / f"{name}.bin"
    values = np.fromfile(path, dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise RuntimeError(
            f"{name} dump size {values.size} != expected {np.prod(shape)}"
        )
    return values.reshape(shape)


def compare(name, actual, expected):
    diff = actual - expected
    rms = float(np.sqrt(np.mean(diff * diff)))
    expected_rms = float(np.sqrt(np.mean(expected * expected)))
    nrmse = rms / max(expected_rms, np.finfo(np.float32).tiny)
    max_abs = float(np.max(np.abs(diff)))
    print(
        name,
        "shape",
        actual.shape,
        "max_abs",
        max_abs,
        "mean_abs",
        float(np.mean(np.abs(diff))),
        "nrmse",
        nrmse,
    )
    if max_abs > 2e-5 or nrmse > 1e-6:
        raise RuntimeError(f"{name} intermediate gradient mismatch")


def main():
    os.makedirs(DUMP_DIR, exist_ok=True)
    os.environ["MUSA_FITTING_DUMP_DIR"] = DUMP_DIR
    tf.load_op_library(fitting.CPU_OP_LIB)
    tf.load_op_library(fitting.MUSA_OP_LIB)
    ops = tensorflow_musa.get_musa_ops()

    reference = build_reference_graph()
    with tf.compat.v1.Session(graph=reference["graph"]) as sess:
        fixture = fitting.run_reference(
            sess,
            reference["graph_tensors"],
            reference["fitting_tensors"],
            reference["identity_pairs"],
            reference["reference"],
            reference["grad_reference"],
            20261027,
        )
    arrays = fixture["arrays"]
    cpu = fitting.numpy_fitting_intermediates(arrays)

    graph = tf.Graph()
    with graph.as_default():
        placeholders = [
            tf.compat.v1.placeholder(tf.float32, shape=array.shape)
            for array in arrays
        ]
        actual = ops.musa_manual_graph_full_fitting(*placeholders)
    with tf.compat.v1.Session(graph=graph) as sess:
        # The second call is a cache hit; its dump files are the validation
        # input so stale miss buffers cannot satisfy this test.
        sess.run(actual, feed_dict=dict(zip(placeholders, arrays)))
        sess.run(actual, feed_dict=dict(zip(placeholders, arrays)))

    compare("g3", read_dump("g3", (240,)), cpu["g3"])
    compare(
        "gt2",
        read_dump("gt2", (1536, 240)),
        cpu["gt2"],
    )
    compare(
        "g2",
        read_dump("g2", (1536, 240)),
        cpu["g2"],
    )
    compare(
        "gt1",
        read_dump("gt1", (1536, 240)),
        cpu["gt1"],
    )
    compare(
        "g1",
        read_dump("g1", (1536, 240)),
        cpu["g1"],
    )
    compare(
        "gt0",
        read_dump("gt0", (1536, 240)),
        cpu["gt0"],
    )
    compare(
        "gx",
        read_dump("gx", (1536, 1600)),
        cpu["gx"],
    )
    compare(
        "energy",
        read_dump("energy", (1536, 1)),
        cpu["energy"],
    )
    print("MANUAL_BACKWARD_INTERMEDIATES_PASS")


if __name__ == "__main__":
    main()
