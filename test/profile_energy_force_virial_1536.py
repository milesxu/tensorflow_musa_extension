import json
import time

import numpy as np
import tensorflow as tf
import tensorflow_musa


PB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/static-shape-probe/model_static_1536_1rank.pb"
CPU_OP_LIB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/install-deepmd/lib/libdeepmd_op.so"
MUSA_OP_LIB = "/data/mingxu/results/ningde_v8_20261021/tf-musa-mpi-v8-20261027/install-deepmd/lib/libdeepmd_op_musa.so"
NLOC = 1536
NALL = 6787


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


def summarize_metadata(metadata, graph):
    op_by_node = {op.name: op.type for op in graph.get_operations()}
    by_type = {}
    nodes = []
    for device_stats in metadata.step_stats.dev_stats:
        for node in device_stats.node_stats:
            duration_us = (
                node.op_end_rel_nanos - node.op_start_rel_nanos
            ) / 1000.0
            op_type = op_by_node.get(node.node_name, "<unknown>")
            by_type.setdefault(op_type, {"count": 0, "sum_us": 0.0})
            by_type[op_type]["count"] += 1
            by_type[op_type]["sum_us"] += duration_us
            nodes.append(
                {
                    "node": node.node_name,
                    "op": op_type,
                    "device": device_stats.device,
                    "duration_us": duration_us,
                }
            )
    by_type = {
        key: {
            "count": value["count"],
            "sum_us": round(value["sum_us"], 3),
        }
        for key, value in sorted(
            by_type.items(), key=lambda item: item[1]["sum_us"], reverse=True
        )
    }
    top_nodes = sorted(
        nodes, key=lambda item: item["duration_us"], reverse=True
    )[:40]
    for node in top_nodes:
        node["duration_us"] = round(node["duration_us"], 3)
    return {
        "node_count": len(nodes),
        "node_time_sum_us": round(
            sum(node["duration_us"] for node in nodes), 3
        ),
        "by_op_type": by_type,
        "top_nodes": top_nodes,
    }


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
            energy = graph.get_tensor_by_name("o_energy:0")
            force = graph.get_tensor_by_name("o_force:0")
            virial = graph.get_tensor_by_name("o_virial:0")

    feeds = make_inputs(20261027)
    fetches = [energy, force, virial]
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

    energy_value, force_value, virial_value = [
        np.asarray(value) for value in values
    ]
    summary = {
        "energy": energy_value.tolist(),
        "force_shape": list(force_value.shape),
        "virial_shape": list(virial_value.shape),
        "force_finite": bool(np.all(np.isfinite(force_value))),
        "virial_finite": bool(np.all(np.isfinite(virial_value))),
        "force_abs_max": float(np.max(np.abs(force_value))),
        "virial_abs_max": float(np.max(np.abs(virial_value))),
        "wall_ms": {
            "mean": float(np.mean(timings)),
            "median": float(np.median(timings)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
        },
        "traced_wall_ms": traced_wall_ms,
        "trace": summarize_metadata(metadata, graph),
    }
    print(json.dumps(summary, indent=2))
    print("PROFILE_ENERGY_FORCE_VIRIAL_PASS")


if __name__ == "__main__":
    main()
