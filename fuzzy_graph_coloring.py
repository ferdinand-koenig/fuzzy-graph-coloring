__version__ = "0.1.1"

import copy
import datetime
import itertools
import math
import random
from collections import Counter
from typing import Tuple, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygad
from numpy.random import default_rng


def _y_ij(i: int, j: int, chromosome) -> bool:
    """
    Function to determine if vertices i and j are assigned color.
    :param i: Vertex
    :param j: Vertex
    :param chromosome: Color assignment
    :return: Boolean
    """
    return chromosome[i] == chromosome[j]


def _fitness_function_factory(graph: nx.Graph):
    """
    Factory to generate fitness-function. Loosely couples graph-instance.
    :param graph: graph-instance (nx.Graph)
    :return: fitness_function
    """

    def _fitness_function(solution: tuple, solution_idx: Any):
        """
        Fitness function to measure the quality of a given chromosome, i.e., solution.
        :param solution: Chromosome: tuple with length equal to number of vertices. Each item, i.e., gen is a color
        :param solution_idx: Required by PyGAD interface but is not used
        :return: fitness: 1 - (Degree of Total Incompatibility (DTI))
        """
        return _get_coloring_score(graph, solution)

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
                    if _y_ij(i - 1, j - 1, parent1):  # if incompatible
                        incompatible_colors_parent1.append(
                            parent1[i - 1])  # add the color to list of incompatible colors
                    if _y_ij(i - 1, j - 1, parent2):
                        incompatible_colors_parent2.append(parent2[i - 1])

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
    """
    Implements a local search procedure. Depends on the local_search_probability parameter.
    :param chromosome: Color assignment
    :param ga_instance: Instance of the Genetic Algorithm
    :return: Returns the color assignment with the best fitness as result
    """
    k = np.max(chromosome)
    best = ([], 0)
    for idx in range(len(chromosome)):
        if (chromosome == chromosome[idx]).sum() == 1:
            continue
        assert (chromosome == chromosome[idx]).sum() != 0
        temp_chromosome = chromosome.copy()
        for color in range(1, k + 1):
            temp_chromosome[idx] = color
            temp_fitness = ga_instance.fitness_func(temp_chromosome, 0)
            if temp_fitness > best[1]:
                best = (temp_chromosome, temp_fitness)
    return best[0]


def _on_generation(ga_instance):
    """
    Genetic Algorithm callback before the end of each generation.
    :param ga_instance: Instance of the Genetic Algorithm
    :return: None
    """
    if ga_instance.local_search_probability > 0:
        for idx, chromosome in enumerate(ga_instance.population):
            if random.random() <= ga_instance.local_search_probability:
                new_chromosome = _local_search(chromosome, ga_instance)
                ga_instance.population[idx] = new_chromosome
    if ga_instance.verbose and ga_instance.generations_completed == 1:
        first_gen_time = datetime.datetime.now() - ga_instance.instance_start_time
        _log(f"First generation took {str(first_gen_time)[2:-4]}")


def _on_start(ga_instance):
    """
    Genetic Algorithm callback at the start of each generation.
    :param ga_instance: Instance of the Genetic Algorithm
    :return: None
    """
    ga_instance.instance_start_time = datetime.datetime.now()


def _on_stop(ga_instance, last_population_fitness):
    """
    Genetic Algorithm callback at the end of each generation.
    :param ga_instance: Instance of the Genetic Algorithm
    :param last_population_fitness: Fitness values of the last population
    :return: None
    """
    if ga_instance.verbose:
        total_elapsed_time = datetime.datetime.now() - ga_instance.start_time
        _log(f"Total elapsed time is {str(total_elapsed_time)[2:-4]}")


def alpha_cut(graph: nx.Graph, alpha: float) -> nx.Graph:
    """
    Alpha-cut for a given NetworkX Graph. Needs attribute "weight" on edges and preserves unconnected vertices.

    :param graph: NetworkX Graph which edges have an attribute "weight"
    :param alpha: Float number for alpha-cut
    :return: Alpha-cut graph
    """
    g = copy.deepcopy(graph)
    for u, v, a in graph.edges(data=True):
        if a["weight"] < alpha:
            g.remove_edge(u, v)
    return g


def _get_coloring_score(graph: nx.Graph, coloring) -> float:
    """
    Calculates the score for a given graph and coloring.
    Coloring can either be a tuple of colors or a dict (node: color assignment).
    :param graph: NetworkX Graph
    :param coloring: Node coloring
    :return: Coloring score (1 - degree of total incompatibility)
    """
    total_incompatibility = 0
    for (i, j) in graph.edges():
        y_ij = _y_ij(i, j, coloring) if isinstance(coloring, dict) else _y_ij(i - 1, j - 1, coloring)  # eq. (2.10a)
        total_incompatibility += graph[i][j]["weight"] * y_ij
    score = 1 - (total_incompatibility / graph.size(weight="weight"))  # 1 - DTI
    return score


def greedy_k_color(graph: nx.Graph, k: int, fair: bool = False) -> dict:
    """
    Greedy algorithm to find a k-coloring for a given graph.
    If fair, Chooses available colors by least frequency of occurrence.
    If not fair, the NetworkX greedy color with the strategy largest_first is used as a basis. This is extended \
        by selecting the most used color and divide it 50:50 with a new color. \
        That is repeated until all colors are used \
    Raises NoSolutionException if the algorithm can not find coloring for the given k.

    :param graph: NetworkX graph
    :param k: Number of colors
    :param fair: Tries to assign colors equitably. Caution: There might be solutions with fair = False but no solution \
        with fair = True
    :return: Color assignment
    """
    if k > graph.number_of_nodes():
        raise InvalidKColoringError(f"Graph has no {k}-coloring as it only has {graph.number_of_nodes()} vertices")
    coloring = {}
    available_colors = {c: 0 for c in range(k)}
    nodes = sorted(graph, key=graph.degree, reverse=True)
    if fair:
        for u in nodes:
            # Set to keep track of colors of neighbours
            neighbour_colors = {coloring[v] for v in graph[u] if v in coloring}
            for color in dict(sorted(available_colors.items(), key=lambda item: item[1])):
                if color not in neighbour_colors:
                    available_colors[color] = available_colors[color] + 1
                    break
            else:
                raise NoSolutionException("No more colors")
            # Assign the new color to the current node.
            coloring[u] = color
    else:
        coloring = nx.greedy_color(graph)
        if max(coloring.values()) + 1 > k:
            raise NoSolutionException(f"Minimal solution needs more colors than k={k} < {max(coloring.values()) + 1}")

        for c in range(max(coloring.values()) + 1, k):
            color_dist = Counter(coloring.values())
            most_used_color = max(color_dist, key=color_dist.get)
            if color_dist[most_used_color] <= 1:
                raise NoSolutionException("Not possible to replace colors")
            nodes_with_most_used_color = [k for k, v in coloring.items() if v == most_used_color]
            replace_color_nodes = default_rng().choice(nodes_with_most_used_color,
                                                       size=math.floor(color_dist[most_used_color] / 2),
                                                       replace=False)
            for node in replace_color_nodes:
                coloring[node] = c

    return coloring


def alpha_fuzzy_color(graph: nx.Graph, k: int, return_alpha: bool = False, fair: bool = False):
    """
    A fuzzy coloring algorithm based on alpha-cuts and greedy coloring.

    :param graph: A networkX graph
    :param k: Number of colors for a k-coloring
    :param return_alpha: Returns the best alpha if set to True (defaults to False)
    :param fair: Parameter for underlying greedy_k_color-function \
        Tries to assign colors equitably. Caution: There might be solutions with fair = False but no solution \
        with fair = True
    :return: Tuple(coloring, score, Optional[alpha])
    """
    if not 1 <= k <= graph.number_of_nodes():
        raise InvalidKColoringError()
    if not is_fuzzy_graph(graph):
        graph = transform_to_fuzzy_graph(graph)

    if k == graph.number_of_nodes():
        return (
            {list(graph.nodes())[c]: c for c in range(graph.number_of_nodes())},
            1
        )

    # Coloring with alpha = 1 alpha-cut (Does a solution exist?)
    latest_alpha = 1
    try:
        coloring = greedy_k_color(alpha_cut(graph, 1), k=k, fair=fair)
    except NoSolutionException:
        raise NoSolutionException("There is no solution where no constraint with weight = 1 is violated!")

    # Coloring with alpha = 0 alpha-cut (Is there a solution without violations)?
    try:
        coloring = greedy_k_color(alpha_cut(graph, 0), k=k, fair=fair)
    except NoSolutionException:
        weights = sorted(set(nx.get_edge_attributes(graph, "weight").values()))

        # binary search on best alpha
        low_idx = 0
        # if 1 is the highest value, do not check it twice ==> high_idx is set one lower
        high_idx = len(weights) - 1 if 1 not in weights else len(weights) - 2

        # modified do while with break condition: low == alpha == high
        while True:
            alpha_idx = low_idx + (high_idx - low_idx) // 2
            try:
                coloring = greedy_k_color(alpha_cut(graph, weights[alpha_idx]), k=k, fair=fair)
            except NoSolutionException:
                if low_idx == alpha_idx and alpha_idx == high_idx:
                    break
                low_idx = alpha_idx + 1
            else:
                latest_alpha = weights[alpha_idx]
                if low_idx == alpha_idx and alpha_idx == high_idx:
                    break
                high_idx = alpha_idx
    else:
        latest_alpha = 0

    if return_alpha:
        return coloring, _get_coloring_score(graph, coloring), latest_alpha
    else:
        print(latest_alpha)
        return coloring, _get_coloring_score(graph, coloring)
    # improvement: if there are lots of constraints with same weight


def genetic_fuzzy_color(graph: nx.Graph, k: int = None, verbose: bool = False, local_search_probability: float = 0.2,
                        crossover_probability: float = 0.8, mutation_probability: float = 0.3,
                        num_generations: int = 15, solutions_per_pop: int = 100) -> dict:
    """
    Calculates the fuzzy coloring of a graph with fuzzy edges by leveraging genetic algorithms.
    Selected parameters can be adjusted. In this context, a chromosome is a feasible solution.

    :param graph: A NetworkX graph
    :param k: Defaults to None. Then, all possible colorings are calculated. \
        If an integer is given, the k-coloring is calculated and returned.
    :param verbose: Gives additional information in console.
    :param local_search_probability: The probability of the execution of a local search. Searches the local space \
        around chromosome for a better solution. Takes longer with higher probability but yields better results.
    :param crossover_probability: The probability of Crossover of two parent chromosomes. Opposite case is to copy \
        the parents into the offspring.
    :param mutation_probability: Gives the probability of mutating a chromosome
    :param num_generations: How many generations will be used to find optimal coloring
    :param solutions_per_pop: How many chromosomes exist per generation
    :return: Returns a dictionary with the keys 'coloring' and 'score' which are the mapping from nodes to colors and \
        the associated fitness or quality. \
        If k is not set, a nested dictionary with the extra level of keys k in (1, ..., n [number of nodes]) is returned
    """
    if k is not None:
        if k > graph.number_of_nodes():
            raise InvalidKColoringError(f"k={k} is bigger than the number of nodes ({graph.number_of_nodes()}).")

    if not is_fuzzy_graph(graph):
        graph = transform_to_fuzzy_graph(graph)
    graph, mapping = _relabel_input_graph(graph)

    colorings = {
        1: (
            {mapping.get(c): 1 for c in range(1, graph.number_of_nodes() + 1)},
            0
        ),
        graph.number_of_nodes(): (
            {mapping.get(c): c for c in range(1, graph.number_of_nodes() + 1)},
            1
        )
    }

    if k == 1 or k == graph.number_of_nodes():
        return colorings[k]

    start_time = datetime.datetime.now()

    num_generations = num_generations
    # solutions_per_pop = offspring_size + keep_parents
    keep_parents = int(solutions_per_pop / 2)
    num_parents_mating = solutions_per_pop - keep_parents
    num_genes = graph.number_of_nodes()
    gene_type = int
    parent_selection_type = "tournament"
    K_tournament = 10
    crossover_type = _incompatibility_elimination_crossover_factory(graph)
    crossover_probability = crossover_probability
    mutation_type = _color_transposition_mutation
    mutation_probability = mutation_probability

    if verbose:
        _log(f"Input graph has {graph.number_of_nodes()} vertices and {graph.number_of_edges()} edges")
        _log("Genetic Algorithm parameters:")
        _log(f"num_generations = {num_generations}")
        _log(f"solutions_per_pop = {solutions_per_pop}")
        _log(f"crossover_probability = {crossover_probability}")
        _log(f"mutation_probability = {mutation_probability}")
        _log(f"local_search_probability = {local_search_probability}")

    for _k in (range(2, graph.number_of_nodes()) if k is None else [k]):
        if verbose:
            _log(f"Starting Genetic Algorithm for k = {_k}")
        initial_population = _initial_population_generator(_k if _k is not None else graph.number_of_nodes(),
                                                           solutions_per_pop,
                                                           num_genes)
        gene_space = {'low': 1, 'high': _k if _k is not None else graph.number_of_nodes()}

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
                               on_generation=_on_generation,
                               on_start=_on_start,
                               on_stop=_on_stop,
                               stop_criteria=["reach_1"])

        ga_instance.local_search_probability = local_search_probability
        ga_instance.verbose = verbose
        ga_instance.start_time = start_time
        ga_instance.run()

        final_solution_fitness = np.max(ga_instance.best_solutions_fitness)
        final_solution_idx = np.argmax(ga_instance.best_solutions_fitness)

        ga_result = (
            {mapping.get(idx + 1): val for idx, val in enumerate(ga_instance.best_solutions[final_solution_idx])},
            final_solution_fitness
        )
        if k is None:
            colorings[_k] = ga_result
        else:
            colorings = ga_result
    return colorings


def bruteforce_fuzzy_color(graph: nx.Graph) -> dict:
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


def draw_weighted_graph(graph: nx.Graph, node_colors=None):
    """
    Plots a given NetworkX graph and labels edges according to their assigned weight.
    Different colors are limited to 10.

    :param graph: NetworkX graph
    :param node_colors: Array of node colors as integer. Actual colors are given by the predefined colormap.
    :return: None
    """
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, labels={node: node for node in graph.nodes()}, node_color=node_colors, cmap=plt.cm.tab10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.show()


def _generate_fuzzy_graph(vertices: int, edge_probability: float, seed: int = None) -> nx.Graph:
    """
    Generates a random fuzzy graph. Using the same seed produces the same graph and weights each time.
    :param vertices: Number of vertices
    :param edge_probability: Probability for edge creation
    :param seed: Random seed.
    :return: NetworkX Graph
    """
    random_graph = nx.fast_gnp_random_graph(n=vertices, p=edge_probability, seed=seed)
    rng = default_rng(seed)
    weights = {edge: np.around(rng.uniform(), decimals=2) for edge in random_graph.edges()}
    nx.set_edge_attributes(random_graph, values=weights, name="weight")
    return random_graph


def is_fuzzy_graph(graph: nx.Graph) -> bool:
    """
    Check if edges have the weight attribute and hold a numeric value < 0 and >= 1.

    :param graph: NetworkX graph
    :return: Bool: graph is fuzzy
    """
    weights = nx.get_edge_attributes(graph, "weight")
    if not weights or len(weights) < graph.number_of_nodes():
        return False
    else:
        for weight in weights.values():
            if not (0 < weight <= 1):
                print(weight)
                return False
    return True


def transform_to_fuzzy_graph(input_graph: nx.Graph) -> nx.Graph:
    """
    Transforms an input graph to a fuzzy graph by setting the edge attribute weight.
    Crisp edges have the weight 1. If some edges already have a weight attribute,
    the weight 1 is added to all remaining crisp edges.
    :param input_graph: NetworkX graph
    :return: NetworkX graph with a valid edge attribute weight
    """
    graph = copy.deepcopy(input_graph)
    weights = nx.get_edge_attributes(graph, "weight")
    if not weights:
        nx.set_edge_attributes(graph, 1, "weight")
    else:
        for (u, v) in graph.edges():
            try:
                weight = graph[u][v]["weight"]
                if not 0 < weight <= 1:
                    raise InvalidFuzzyGraphError(
                        f"Input graph has invalid weight attribute {weight} for edge ({u},{v})")
            except KeyError:
                graph[u][v]["weight"] = 1
    return graph


def _relabel_input_graph(graph: nx.Graph) -> Tuple[nx.Graph, dict]:
    """
    Relabels an input graph to use consecutive integers as node labels.
    Returns a copy and the new mapping which can be used to revert the relabeling.

    :param graph: NetworkX graph
    :return: Relabeled Graph, mapping
    """
    mapping = {new + 1: old for new, old in enumerate(graph.nodes())}
    int_node_graph = nx.relabel_nodes(graph, mapping={old: new + 1 for new, old in enumerate(graph.nodes())}, copy=True)
    return int_node_graph, mapping


def _build_example_graph_1() -> nx.Graph:
    """
    Build example fuzzy graph also presented in the paper Fig. 2.2
    :return: NetworkX Graph
    """
    tg1 = nx.Graph()
    tg1.add_edge(1, 2, weight=0.7)
    tg1.add_edge(1, 3, weight=0.8)
    tg1.add_edge(1, 4, weight=0.5)
    tg1.add_edge(2, 3, weight=0.3)
    tg1.add_edge(2, 4, weight=0.4)
    tg1.add_edge(3, 4, weight=1.0)
    return tg1


def _build_example_graph_2() -> nx.Graph:
    """
    Build example fuzzy graph
    :return: NetworkX Graph
    """
    tg2 = nx.Graph()
    tg2.add_edge(1, 2, weight=0.4)
    tg2.add_edge(1, 3, weight=0.7)
    tg2.add_edge(1, 4, weight=0.8)
    tg2.add_edge(2, 4, weight=0.2)
    tg2.add_edge(2, 5, weight=0.9)
    tg2.add_edge(3, 4, weight=0.3)
    tg2.add_edge(3, 6, weight=1.0)
    tg2.add_edge(4, 5, weight=0.3)
    tg2.add_edge(4, 6, weight=0.5)
    tg2.add_edge(5, 6, weight=0.7)
    tg2.add_edge(5, 7, weight=0.8)
    tg2.add_edge(5, 8, weight=0.5)
    tg2.add_edge(6, 7, weight=0.7)
    tg2.add_edge(7, 8, weight=0.6)
    return tg2


def _build_example_crisp_graph() -> nx.Graph:
    """
    Build example crisp graph with the chromatic number 5.
    :return: NetworkX Graph
    """
    cg = nx.Graph()
    cg.add_edge(1, 2)
    cg.add_edge(1, 3)
    cg.add_edge(1, 5)
    cg.add_edge(1, 6)
    cg.add_edge(1, 7)
    cg.add_edge(2, 3)
    cg.add_edge(2, 5)
    cg.add_edge(2, 6)
    cg.add_edge(3, 4)
    cg.add_edge(3, 5)
    cg.add_edge(3, 6)
    cg.add_edge(4, 6)
    cg.add_edge(4, 10)
    cg.add_edge(5, 6)
    cg.add_edge(5, 8)
    cg.add_edge(5, 9)
    cg.add_edge(6, 8)
    cg.add_edge(7, 8)
    cg.add_edge(8, 10)
    cg.add_edge(9, 10)
    return cg


def _log(message: str):
    """
    Print message with timestamp
    :param message: Message to print
    :return: None
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")


class InvalidKColoringError(Exception):
    """Raised when the k-coloring does not exist"""
    pass


class InvalidFuzzyGraphError(Exception):
    """Raised when a fuzzy graph is invalid. For example, weight > 1."""
    pass


class NoSolutionException(Exception):
    """Raised when no solution is found"""
    pass


if __name__ == '__main__':
    graph = _build_example_graph_2()
    # coloring, score, alpha = alpha_fuzzy_color(graph, 3, return_alpha=True, fair=True)
    # print(score, alpha, coloring)
    # draw_weighted_graph(graph, [coloring.get(node) for node in graph])

    # nx_coloring = nx.greedy_color(graph)
    # print(nx_coloring)
    # draw_weighted_graph(graph, [nx_coloring.get(node) for node in graph])
    # print(max(nx_coloring.values())+1)

    coloring, score, alpha = alpha_fuzzy_color(graph, 3, return_alpha=True, fair=True)
    print(coloring, score, alpha)
    draw_weighted_graph(graph, [coloring.get(node) for node in graph])
    print(max(coloring.values()) + 1)

    # coloring, score = genetic_fuzzy_color(graph, 3)
    # print(score, coloring)
    # draw_weighted_graph(graph, [coloring.get(node) for node in graph])
