import networkx as nx
import numpy as np
from numpy.random import default_rng

from fuzzy_graph_coloring import __version__, fuzzy_color


def test_version():
    assert __version__ == '0.1.0'


def _generate_fuzzy_graph(vertices: int, edge_probability: float, seed: int) -> nx.Graph:
    random_graph = nx.fast_gnp_random_graph(n=vertices, p=edge_probability, seed=seed)
    rng = default_rng(seed)
    weight_values = np.around(rng.uniform(size=random_graph.number_of_edges()), decimals=2)
    weights = {edge: weight_values for edge in random_graph.edges()}
    nx.set_edge_attributes(random_graph, values=weights, name="weight")
    return random_graph


def _build_example_graph_1() -> nx.Graph:
    """
    Build example fuzzy graph also presented in the paper Fig. 2.2
    :return: NetworkX Graph
    """
    TG1 = nx.Graph()
    TG1.add_edge(1, 2, weight=0.7)
    TG1.add_edge(1, 3, weight=0.8)
    TG1.add_edge(1, 4, weight=0.5)
    TG1.add_edge(2, 3, weight=0.3)
    TG1.add_edge(2, 4, weight=0.4)
    TG1.add_edge(3, 4, weight=1.0)
    return TG1


def test_fuzzy_color_case_1():
    test_graph = _build_example_graph_1()
    solution = fuzzy_color(test_graph)
    expected_solution = {
        1: {
            "coloring": {
                1: 1,
                2: 1,
                3: 1,
                4: 1
            },
            "score": 0
        },
        2: {
            "coloring": {
                1: 1,
                2: 2,
                3: 1,
                4: 2
            },
            "score": 0.7837
        },
        3: {
            "coloring": {
                1: 1,
                2: 2,
                3: 2,
                4: 3
            },
            "score": 0.9189
        },
        4: {
            "coloring": {
                1: 1,
                2: 2,
                3: 3,
                4: 4
            },
            "score": 1
        }
    }
    assert solution == expected_solution
