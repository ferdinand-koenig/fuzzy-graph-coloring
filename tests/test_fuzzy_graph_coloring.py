import networkx as nx
import pytest

from fuzzy_graph_coloring import (
    __version__,
    genetic_fuzzy_color,
    _build_example_crisp_graph,
    _build_example_graph_1,
    _build_example_graph_2
)


def test_version():
    assert __version__ == '0.1.0'


def test_fuzzy_color_edge_case_1():
    test_graph = _build_example_crisp_graph()
    nx.set_edge_attributes(test_graph, 1, "weight")  # set all weights to 1
    solution = genetic_fuzzy_color(test_graph)
    expected_scores = {
        1: 0,
        2: 0.8,
        3: 0.9,
        4: 0.95,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
    }
    for k, score in expected_scores.items():
        assert solution[k][1] >= score, f"{k}-coloring is not the minimal coloring"


def test_fuzzy_color_edge_case_2():
    test_graph = _build_example_crisp_graph()
    nx.set_edge_attributes(test_graph, 0, "weight")  # set all weights to 0
    with pytest.raises(Exception):  # Exception is expected as weight = 0 is invalid
        _ = genetic_fuzzy_color(test_graph)


def test_fuzzy_color_edge_case_3():
    test_graph = _build_example_crisp_graph()
    nx.set_edge_attributes(test_graph, 0.5, "weight")  # set all weights to 0.5
    solution = genetic_fuzzy_color(test_graph)
    expected_scores = {
        1: 0,
        2: 0.8,
        3: 0.9,
        4: 0.95,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
    }
    for k, score in expected_scores.items():
        assert solution[k][1] >= score, f"{k}-coloring is not the minimal coloring"


def test_fuzzy_color_case_1():
    test_graph = _build_example_graph_1()
    solution = genetic_fuzzy_color(test_graph)
    expected_scores = {
        1: 0,
        2: 0.7837,
        3: 0.9189,
        4: 1
    }
    for k, score in expected_scores.items():
        assert solution[k][1] >= score, f"{k}-coloring is not the minimal coloring"


def test_fuzzy_color_case_2():
    test_graph = _build_example_graph_2()
    solution = genetic_fuzzy_color(test_graph)
    expected_scores = {
        1: 0,
        2: 0.7976,
        3: 0.9761,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1
    }
    for k, score in expected_scores.items():
        assert solution[k][1] >= score, f"{k}-coloring is not the minimal coloring"
