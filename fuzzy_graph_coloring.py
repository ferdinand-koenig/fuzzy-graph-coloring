__version__ = "0.1.0"

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import default_rng


def main():
    print("Hello World")


def fuzzy_color(graph: nx.Graph):
    return {}


def bruteforce_fuzzy_color(graph: nx.Graph):
    """
    Finds the minimal k-coloring for a fuzzy graph by bruteforce all possible color assignments.
    Excludes the cases k=1 and k=number of nodes since the degree of total incompatibility is 1 and 0.

    :param graph: Fuzzy NetworkX graph
    :return: dict containing minimal k-colorings (excluding k=1 and k=n)
    """
    colors = range(1, graph.number_of_nodes() + 1)
    color_assignments = itertools.product(*[colors] * graph.number_of_nodes())
    weight_sum = graph.size(weight="weight")
    colorings = {}
    for color_assignment in color_assignments:
        if len(set(color_assignment)) in [1, graph.number_of_nodes()]:
            continue
        total_incompatibility = 0
        for color in colors:
            nodes_with_color = [index + 1 for index, element in enumerate(color_assignment) if element == color]
            incompatible_edges = [(u, v) for (u, v) in graph.edges() if u in nodes_with_color and v in nodes_with_color]
            for ie in incompatible_edges:
                total_incompatibility += graph[ie[0]][ie[1]]["weight"]
        k = len(set(color_assignment))
        score = 1 - (total_incompatibility / weight_sum)
        if k not in colorings.keys():
            colorings[k] = {}
            colorings[k]["coloring"] = {c: color_assignment[c - 1] for c in colors}
            colorings[k]["score"] = score
        elif score > colorings[k]["score"]:
            colorings[k]["coloring"] = {c: color_assignment[c - 1] for c in colors}
            colorings[k]["score"] = score
    return colorings


def draw_weighted_graph(graph: nx.Graph):
    """
    Plots a given NetworkX graph and labels edges according to their assigned weight.

    :param graph: NetworkX graph
    :return: None
    """
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, labels={node: node for node in graph.nodes()})
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.show()


def _generate_fuzzy_graph(vertices: int, edge_probability: float, seed: int) -> nx.Graph:
    random_graph = nx.fast_gnp_random_graph(n=vertices, p=edge_probability, seed=seed)
    rng = default_rng(seed)
    weights = {edge: np.around(rng.uniform(), decimals=2) for edge in random_graph.edges()}
    nx.set_edge_attributes(random_graph, values=weights, name="weight")
    return random_graph


def is_fuzzy_graph(graph: nx.Graph) -> bool:
    """
    Check if edges have the weight attribute and hold numeric value < 0 and >= 1.

    :param graph: NetworkX graph
    :return: Bool if graph is fuzzy
    """
    weights = nx.get_edge_attributes(graph, "weight")
    if not weights:
        return False
    else:
        for weight in weights.values():
            if not (0 < weight <= 1):
                print(weight)
                return False
    return True


if __name__ == '__main__':
    main()
