import argparse

from tensorflow.core.framework import graph_pb2


STALE = {
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
LAMMPS_OUTPUTS = [
    "o_energy",
    "o_force",
    "o_atom_energy",
    "o_atom_virial",
]


def ancestors(nodes, output):
    seen = set()
    stack = [output]
    while stack:
        name = stack.pop().split(":")[0].lstrip("^")
        if name in seen or name not in nodes:
            continue
        seen.add(name)
        stack.extend(nodes[name].input)
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_def")
    args = parser.parse_args()
    graph = graph_pb2.GraphDef()
    with open(args.graph_def, "rb") as handle:
        graph.ParseFromString(handle.read())
    nodes = {node.name: node for node in graph.node}
    print(f"graph_nodes={len(graph.node)}")
    for output in LAMMPS_OUTPUTS:
        closure = ancestors(nodes, output)
        print(
            f"{output} nodes={len(closure)} "
            f"manual={'manual_full_fitting_type_0' in closure} "
            f"stale={sorted(STALE & closure)}"
        )


if __name__ == "__main__":
    main()
