__version__ = "0.1.0"

import datetime
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
    return chromosome[i - 1] == chromosome[j - 1]


def _fitness_function_factory(graph: nx.Graph):
    """
    Factory to generate fitness-function. Loosely couples graph-instance.
    :param graph: graph-instance (nx.Graph)
    :return: fitness_function
    """

    def _fitness_function(solution: tuple, solution_idx):
        """
        Fitness function to measure the quality of a given chromosome, i.e., solution.
        :param solution: Chromosome: tuple with length equal to number of vertices. Each item, i.e., gen is a color
        :param solution_idx:
        :return: fitness: 1 - (Degree of Total Incompatibility (DTI))
        """
        total_incompatibility = 0
        for (i, j) in graph.edges():
            total_incompatibility += graph[i][j]["weight"] * _y_ij(i, j, solution)  # eq. (2.10a)
        fitness = 1 - (total_incompatibility / graph.size(weight="weight"))  # 1 - DTI
        return fitness

    return _fitness_function


def _incompatibility_elimination_crossover_factory(graph: nx.Graph):
    """
    Factory to generate IEX-function. Loosely couples graph-instance.
    :param graph: graph-instance (nx.Graph)
    :return: incompatibility_elimination_crossover (IEX)-function
    """

    def _incompatibility_elimination_crossover(parents, offspring_size, ga_instance):
        """
        IEX-function.
        :param parents: The selected parents.
        :param offspring_size: The size of the offspring as a tuple of 2 numbers: (the offspring size, number of genes)
        :param ga_instance: Instance of the pygad.GA class
        :return: offspring
        """
        # assert len(parents) == offspring_size[0]
        # assert offspring_size[0] == ga_instance.sol_per_pop

        idx = 0
        offspring = np.empty((0, parents.shape[1]), int)
        while len(offspring) != offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

            child1 = parent1.copy()
            child2 = parent2.copy()

            if random.random() <= ga_instance.crossover_probability and ga_instance.crossover_probability > 0:
                # Get incompatible colors...
                incompatible_colors_parent1 = []
                incompatible_colors_parent2 = []
                for (i, j) in graph.edges():
                    if _y_ij(i, j, parent1):  # if incompatible
                        incompatible_colors_parent1.append(parent1[i])  # add the color to list of incompatible colors
                    if _y_ij(i, j, parent2):
                        incompatible_colors_parent2.append(parent2[i])

                # ... and exchange with the colors of other parent except for a random appearance

                for ic in incompatible_colors_parent1:
                    ic_mask = (child1 == ic)
                    ic_mask[default_rng().choice(np.where(ic_mask)[0])] = False
                    child1[ic_mask] = parent2[ic_mask]
                for ic in incompatible_colors_parent2:
                    ic_mask = (child2 == ic)
                    ic_mask[default_rng().choice(np.where(ic_mask)[0])] = False
                    child2[ic_mask] = parent1[ic_mask]

            offspring = np.append(offspring, [child1], axis=0)
            offspring = np.append(offspring, [child2], axis=0)
            assert np.max(child1) == len(np.unique(child1))
            assert np.max(child2) == len(np.unique(child2))
            idx += 1

        return offspring

    return _incompatibility_elimination_crossover


def _color_transposition_mutation(offspring, ga_instance):
    """
    CTM. Randomly selects two colors and switches them mutually.
    The chance that the mutation operator is applied to a chromosome is given by the mutation probability.
    :param offspring: The offspring to be mutated
    :param ga_instance: Instance of the pygad.GA class
    :return: Mutated offspring
    """
    mutation_offspring = np.empty((0, offspring.shape[1]), int)
    for chromosome in offspring:
        if random.random() <= ga_instance.mutation_probability and ga_instance.mutation_probability > 0:
            # for each chromosome, there is a chance to be mutated
            color_a, color_b = default_rng().choice(np.unique(chromosome), size=2,
                                                    replace=False)  # take 2 different colors
            # swap those colors
            mask_a = (chromosome == color_a)
            mask_b = (chromosome == color_b)
            chromosome[mask_a] = color_b
            chromosome[mask_b] = color_a
        mutation_offspring = np.append(mutation_offspring, [chromosome], axis=0)
        assert np.max(chromosome) == len(np.unique(chromosome))
    return mutation_offspring


def _initial_population_generator(k: int, sol_per_pop: int, num_genes: int):
    """
    Generates the initial Genetic Algorithm population.
    :param k: Number of available colors (k-coloring)
    :param sol_per_pop: Number of solutions/chromosomes per population
    :param num_genes: Number of genes in the solution/chromosome
    :return: Initial population as nested numpy array
    """
    initial_population = np.empty((0, num_genes), int)
    for _ in range(sol_per_pop):
        chromosome = np.zeros(num_genes, int)
        for color, gene_idx in enumerate(random.sample(range(num_genes), k)):
            # random.sample(range(num_genes), k) -> [0,2] # positions of the colors
            # 1: 0, 2: 2 ; color to position mapping
            chromosome[gene_idx] = color + 1
            # [1, 0, 2]
        zero_mask = (chromosome == 0)  # [false, true, false]
        chromosome[zero_mask] = default_rng().choice(range(1, k + 1), zero_mask.sum())
        initial_population = np.append(initial_population, [chromosome], axis=0)
    return initial_population


def _local_search(chromosome: np.array, ga_instance) -> np.array:
    k = np.max(chromosome)
    best = ([], 0)
    for idx in range(len(chromosome)):
        if (chromosome == chromosome[idx]).sum() == 1:
            continue
        assert (chromosome == chromosome[idx]).sum() != 0
        temp_chromosome = chromosome.copy()
        for color in range(1, k+1):
            temp_chromosome[idx] = color
            temp_fitness = ga_instance.fitness_func(temp_chromosome, 0)
            if temp_fitness > best[1]:
                best = (temp_chromosome, temp_fitness)
    return best[0]


def on_generation(ga_instance):
    print("almost done", ga_instance.generations_completed, datetime.datetime.now().strftime("%H:%M:%S.%f"))
    if not ga_instance.local_search_probability > 0:
        return
    for idx, chromosome in enumerate(ga_instance.population):
        if random.random() <= ga_instance.local_search_probability:
            new_chromosome = _local_search(chromosome, ga_instance)
            ga_instance.population[idx] = new_chromosome


def fuzzy_color(graph: nx.Graph, k_coloring: int = None):
    """
    Hallo
    :param graph:
    :param k_coloring:
    :return:
    """
    num_generations = 15
    solutions_per_pop = 100  # solutions_per_pop = offspring_size + keep_parents
    num_parents_mating = solutions_per_pop
    keep_parents = 50
    num_genes = graph.number_of_nodes()
    gene_type = int
    parent_selection_type = "tournament"
    K_tournament = 10
    crossover_type = _incompatibility_elimination_crossover_factory(graph)
    crossover_probability = 0.8
    mutation_type = _color_transposition_mutation
    mutation_probability = 0.3

    if k_coloring is None:
        colorings = {
            1: {
                "coloring": {c: 1 for c in range(1, graph.number_of_nodes() + 1)},
                "score": 0
            },
            graph.number_of_nodes(): {
                "coloring": {c: c for c in range(1, graph.number_of_nodes() + 1)},
                "score": 1
            }
        }
    else:
        colorings = {}

    for k in (range(2, graph.number_of_nodes()) if k_coloring is None else [k_coloring]):
        initial_population = _initial_population_generator(k if k is not None else graph.number_of_nodes(),
                                                           solutions_per_pop,
                                                           num_genes)
        gene_space = {'low': 1, 'high': k if k is not None else graph.number_of_nodes()}

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               keep_parents=keep_parents,
                               initial_population=initial_population,
                               gene_type=gene_type,
                               gene_space=gene_space,
                               fitness_func=_fitness_function_factory(graph),
                               parent_selection_type=parent_selection_type,
                               K_tournament=K_tournament,
                               crossover_type=crossover_type,
                               crossover_probability=crossover_probability,
                               mutation_type=mutation_type,
                               mutation_probability=mutation_probability,
                               save_best_solutions=True,
                               on_generation=on_generation)

        ga_instance.local_search_probability = 0.2
        ga_instance.run()
        # ga_instance.plot_fitness()

        final_solution_fitness = np.max(ga_instance.best_solutions_fitness)
        final_solution_idx = np.argmax(ga_instance.best_solutions_fitness)
        print(ga_instance.best_solutions[final_solution_idx], final_solution_fitness)

        ga_result = {
            "coloring": {idx + 1: val for idx, val in enumerate(ga_instance.best_solutions[final_solution_idx])},
            "score": final_solution_fitness
        }
        if k_coloring is None:
            colorings[k] = ga_result
        else:
            colorings = ga_result
    return colorings


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
    Check if edges have the weight attribute and hold a numeric value < 0 and >= 1.

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


if __name__ == '__main__':
    print(fuzzy_color(_build_example_graph_2(), None))
    # fuzzy_color(_generate_fuzzy_graph(25, 0.25, 42), 3)
