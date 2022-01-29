__version__ = "0.1.0"

import matplotlib.pyplot as plt
import networkx as nx


def main():
    print("Hello World")


def fuzzy_color(graph: nx.Graph):
    return {}


def draw_weighted_graph(graph: nx.Graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, labels={node: node for node in graph.nodes()})
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.show()


if __name__ == '__main__':
    main()
