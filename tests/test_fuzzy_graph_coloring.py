import networkx as nx

from fuzzy_graph_coloring import __version__, fuzzy_color


def test_version():
    assert __version__ == '0.1.0'


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


def _build_example_graph_2() -> nx.Graph:
    TG2 = nx.Graph()
    TG2.add_edge(1, 2, weight=0.4)
    TG2.add_edge(1, 3, weight=0.7)
    TG2.add_edge(1, 4, weight=0.8)
    TG2.add_edge(2, 4, weight=0.2)
    TG2.add_edge(2, 5, weight=0.9)
    TG2.add_edge(3, 4, weight=0.3)
    TG2.add_edge(3, 6, weight=1.0)
    TG2.add_edge(4, 5, weight=0.3)
    TG2.add_edge(4, 6, weight=0.5)
    TG2.add_edge(5, 6, weight=0.7)
    TG2.add_edge(5, 7, weight=0.8)
    TG2.add_edge(5, 8, weight=0.5)
    TG2.add_edge(6, 7, weight=0.7)
    TG2.add_edge(7, 8, weight=0.6)
    return TG2


def _build_example_crisp_graph() -> nx.Graph:
    CG = nx.Graph()
    CG.add_edge(1, 2)
    CG.add_edge(1, 3)
    CG.add_edge(1, 5)
    CG.add_edge(1, 6)
    CG.add_edge(1, 7)
    CG.add_edge(2, 3)
    CG.add_edge(2, 5)
    CG.add_edge(2, 6)
    CG.add_edge(3, 4)
    CG.add_edge(3, 5)
    CG.add_edge(3, 6)
    CG.add_edge(4, 6)
    CG.add_edge(4, 10)
    CG.add_edge(5, 6)
    CG.add_edge(5, 8)
    CG.add_edge(5, 9)
    CG.add_edge(6, 8)
    CG.add_edge(7, 8)
    CG.add_edge(8, 8)
    CG.add_edge(8, 10)
    CG.add_edge(9, 10)
    return CG


def test_fuzzy_color_case_1():
    test_graph = _build_example_graph_1()
    solution = fuzzy_color(test_graph)
    expected_scores = {
        1: 0,
        2: 0.7837,
        3: 0.9189,
        4: 1
    }
    for k, score in expected_scores.items():
        assert solution[k][1] >= score


def test_fuzzy_color_case_2():
    test_graph = _build_example_graph_2()
    solution = fuzzy_color(test_graph)
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
        assert solution[k][1] >= score
