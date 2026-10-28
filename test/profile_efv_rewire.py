import json
import time

import numpy as np
import tensorflow as tf
import tensorflow_musa

import profile_energy_force_virial_1536 as prof


BASELINE_PB = (
    "/data/mingxu/results/ningde_v8_20261021/"
    "tf-musa-mpi-v8-20261027/static-shape-probe/"
    "model_static_1536_1rank.pb"
)
REWIRED_PB = (
    "/data/mingxu/tmp/tf-musa-stage-graph-probe-20261027/"
    "model_static_1536_1rank_manual_efv_full.pb"
)


def import_graph(path):
    graph_def = tf.compat.v1.GraphDef()
    with open(path, "rb") as handle:
        graph_def.ParseFromString(handle.read())
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            tf.import_graph_def(graph_def, name="")
    return graph


def run_profile(label, graph, feeds):
    fetches = [
        graph.get_tensor_by_name("o_energy:0"),
        graph.get_tensor_by_name("o_force:0"),
        graph.get_tensor_by_name("o_virial:0"),
    ]
    with tf.compat.v1.Session(graph=graph) as sess:
        for _ in range(2):
            sess.run(fetches, feed_dict=feeds)
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            sess.run(fetches, feed_dict=feeds)
            timings.append((time.perf_counter() - start) * 1e3)
        options = tf.compat.v1.RunOptions(
            trace_level=tf.compat.v1.RunOptions.FULL_TRACE
        )
        metadata = tf.compat.v1.RunMetadata()
        start = time.perf_counter()
        values = sess.run(
            fetches,
            feed_dict=feeds,
            options=options,
            run_metadata=metadata,
        )
        traced_wall_ms = (time.perf_counter() - start) * 1e3
    summary = prof.summarize_metadata(metadata, graph)
    executed_type0 = []
    for device_stats in metadata.step_stats.dev_stats:
        for node in device_stats.node_stats:
            if "type_0" in node.node_name and any(
                suffix in node.node_name
                for suffix in ("MatMul", "BiasAdd", "Tanh", "Reshape", "/mul")
            ):
                executed_type0.append(node.node_name)
    manual_node_us = None
    for node in summary["top_nodes"]:
        if node["node"] == "manual_full_fitting_type_0":
            manual_node_us = node["duration_us"]
    return {
        "label": label,
        "wall_ms": {
            "mean": float(np.mean(timings)),
            "median": float(np.median(timings)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
        },
        "traced_wall_ms": traced_wall_ms,
        "trace_node_sum_us": summary["node_time_sum_us"],
        "executed_type0_numerical_nodes": executed_type0,
        "manual_node_us": manual_node_us,
        "energy": np.asarray(values[0]).tolist(),
        "force_shape": list(np.asarray(values[1]).shape),
        "virial_shape": list(np.asarray(values[2]).shape),
        "trace": summary,
    }


def main():
    tf.load_op_library(prof.CPU_OP_LIB)
    tf.load_op_library(prof.MUSA_OP_LIB)
    feeds = prof.make_inputs(20261027)
    baseline = run_profile("baseline", import_graph(BASELINE_PB), feeds)
    manual = run_profile("manual", import_graph(REWIRED_PB), feeds)
    baseline_ms = baseline["wall_ms"]["mean"]
    manual_ms = manual["wall_ms"]["mean"]
    result = {
        "baseline": baseline,
        "manual": manual,
        "wall_speedup_baseline_over_manual": baseline_ms / manual_ms,
    }
    print(json.dumps(result, indent=2))
    print("EFV_REWIRE_PROFILE_PASS")


if __name__ == "__main__":
    main()
