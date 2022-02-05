__version__ = "0.1.0"

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygad
import random
from numpy.random import default_rng


def main():
    print("Hello World")


def _y_ij(i: int, j: int, chromosome: tuple) -> bool:
    return chromosome[i] == chromosome[j]


def fitness_function_factory(graph: nx.Graph, k: int):
    """

    :param graph:
    :param k: k-coloring
    :return:
    """

    def fitness_function(solution: tuple, solution_idx):
        total_incompatibility = 0
        for (i, j) in graph.edges():
            total_incompatibility += graph[i][j]["weight"] * _y_ij(i, j, solution)  # eq. (2.10a)
        fitness = 1 - (total_incompatibility / graph.size(weight="weight"))  # 1 - DTI
        return fitness

    return fitness_function


def incompatibility_elimination_crossover_factory(graph: nx.Graph):
    """

    :param graph:
    :return:
    """

    def incompatibility_elimination_crossover(parents, offspring_size, ga_instance):
        """

        :param parents: The selected parents.
        :param offspring_size: The size of the offspring as a tuple of 2 numbers: (the offspring size, number of genes)
        :param ga_instance: Instance of the pygad.GA class
        :return:
        """
        assert parents.size == offspring_size[0]
        assert offspring_size[0] == ga_instance.sol_per_pop

        idx = 0
        offspring = np.array([])
        while len(offspring) != offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            # Get incompatible colors...
            incompatible_colors_parent1 = []
            incompatible_colors_parent2 = []
            for (i, j) in graph.edges():
                if _y_ij(i, j, parent1):  # if incompatible
                    incompatible_colors_parent1.append(parent1[i])  # add the color to list of incompatible colors
                if _y_ij(i, j, parent2):
                    incompatible_colors_parent2.append(parent2[i])

            # ... and exchange with the colors of other parent except for last appearance
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
            for ic in incompatible_colors_parent1:
                ic_mask = (parent1 == ic)
                # ic_mask[len(ic_mask) - 1 - ic_mask[::-1].index(True)] = False
                ic_mask[np.where(ic_mask)[0][-1]] = False
                child1[ic_mask] = parent2[ic_mask]
            for ic in incompatible_colors_parent2:
                ic_mask = (parent2 == ic)
                ic_mask[np.where(ic_mask)[0][-1]] = False
                child2[ic_mask] = parent1[ic_mask]

            offspring = np.append(offspring, child1)
            offspring = np.append(offspring, child2)
            idx += 1

        return offspring

    return incompatibility_elimination_crossover


def color_transposition_mutation(offspring, ga_instance):
    """
    Randomly selects two colors and switches them mutually.
    The chance that the mutation operator is applied to a chromosome is given by the mutation probability.
    :param offspring: The offspring to be mutated
    :param ga_instance: Instance of the pygad.GA class
    :return: Mutated offspring
    """
    mutation_offspring = np.array([])
    for chromosome in offspring:
        if random.random() <= ga_instance.mutation_probability:  # for each chromosome, there is a chance to be mutated
            color_a, color_b = default_rng().choice(np.unique(chromosome), size=2,
                                                    replace=False)  # take 2 different colors
            # swap those colors
            mask_a = (chromosome == color_a)
            mask_b = (chromosome == color_b)
            chromosome[mask_a] = color_b
            chromosome[mask_b] = color_a
        mutation_offspring = np.append(mutation_offspring, chromosome)
    return mutation_offspring


def initial_population_generator(k: int, sol_per_pop: int, num_genes: int):
    initial_population = np.array([])
    for _ in range(sol_per_pop):
        chromosome = np.zeros(num_genes, int)
        for color, gene_idx in enumerate(random.sample(range(num_genes), k)):
            # random.sample(range(num_genes), k) -> [0,2] # positions of the colors
            # 1: 0, 2: 2 ; color to position mapping
            chromosome[gene_idx] = color + 1
            # [1, 0, 2]
        zero_mask = (chromosome == 0)  # [false, true, false]
        chromosome[zero_mask] = default_rng().choice(range(1, k + 1), zero_mask.sum())
        initial_population = np.append(initial_population, chromosome)
    return initial_population


def fuzzy_color(graph: nx.Graph, k: int = None):
    num_generations = 15
    solutions_per_pop = 100
    num_parents_mating = solutions_per_pop
    num_genes = graph.number_of_nodes()
    initial_population = initial_population_generator(k if k is not None else graph.number_of_nodes(),
                                                      solutions_per_pop,
                                                      num_genes)

    gene_type = int
    gene_space = {'low': 1, 'high': k if k is not None else graph.number_of_nodes()}

    parent_selection_type = "tournament"
    K_tournament = 5

    crossover_type = incompatibility_elimination_crossover_factory(graph)
    crossover_probability = 0.8

    mutation_type = color_transposition_mutation
    mutation_probability = 0.3

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           gene_type=gene_type,
                           gene_space=gene_space,
                           fitness_func=fitness_function_factory(graph,
                                                                 k if k is not None else graph.number_of_nodes()),
                           parent_selection_type=parent_selection_type,
                           K_tournament=K_tournament,
                           crossover_type=crossover_type,
                           crossover_probability=crossover_probability,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_probability)

    print("initial population:")
    print(ga_instance.initial_population)

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
        score = 1 - (total_incompatibility / weight_sum)  # 1 - DTI
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
