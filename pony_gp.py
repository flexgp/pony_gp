import collections
import sys
import configparser
import csv
import argparse
import time
import random
import math
import copy
""" Implementation of Genetic Programming(GP), the purpose of this
code is to describe how the algorithm works. The intended use is for
teaching. The design is supposed to be simple, self contained and use
core python libraries.

Genetic Programming
===================
An individual is a dictionary with two keys:
  - *genome* -- A tree
  - *fitness* -- The fitness of the evaluated tree
The fitness is maximized.
The nodes in a GP tree consists of different symbols. The symbols are either
functions (internal nodes with arity > 0) or terminals (leaf nodes with arity
= 0) The symbols is represented as a dictionary with the keys:
  - *arities* -- A dictionary where a key is a symbol and the value is the arity
  - *terminals* -- A list of strings(symbols) with arity 0
  - *functions* -- A list of strings(symbols) with arity > 0

Fitness Function
----------------
Find a symbolic expression (function) which yields the lowest error
for a given set of inputs.
Inputs have explanatory variables that have a corresponding
output. The input data is split into test and training data. The
training data is used to generate symbolic expressions and the test
data is used to evaluate the out-of-sample performance of the
evaluated expressions.

Pony GP Parameters
------------------
The parameters for Pony GP are in a dictionary.

.. codeauthor:: Erik Hemberg <hembergerik@csail.mit.edu>
"""
DEFAULT_FITNESS = -float("inf")


def append_node(node, symbol):
    """
    Return the appended node. Append a symbol to the node.

    :param node: The node that will be appended to
    :type node: list
    :param symbol: The symbol that is appended
    :type symbol: str
    :return: The new node
    :rtype: list
    """

    # Create a list with the symbol and append it to the node
    new_node = [symbol]
    node.append(new_node)

    return new_node


def grow(node, depth, max_depth, full, symbols):
    """
    Recursively grow a node to max depth in a pre-order, i.e. depth-first
    left-to-right traversal.

    :param node: Root node of subtree
    :type node: list
    :param depth: Current tree depth
    :type depth: int
    :param max_depth: Maximum tree depth
    :type max_depth: int
    :param full: grows the tree to max depth when true
    :type full: bool
    :param symbols: set of symbols to chose from
    :type symbols: dict
    """
    # grow is called recursively in the loop. The loop iterates arity number
    # of times. The arity is given by the node symbol
    node_symbol = node[0]
    for _ in range(symbols["arities"][node_symbol]):
        # Get a random symbol
        symbol = get_random_symbol(depth, max_depth, symbols, full)
        # Create a child node and append it to the tree
        new_node = append_node(node, symbol)
        # if statement
        # Call grow with the child node as the current node
        grow(new_node, depth + 1, max_depth, full, symbols)

        assert len(node) == (_ + 2), len(node)

    assert depth <= max_depth, "%d %d" % (depth, max_depth)


def get_children(node):
    """
    Return the children of the node. The children are all the elements of the
    except the first

    :param node: The node
    :type node: list
    :return: The children of the node
    :rtype: list
    """
    # Take a slice of the list except the head
    return node[1:]


def get_number_of_nodes(root, cnt):
    """
    Return the number of nodes in the tree. A recursive depth-first
    left-to-right search is done

    :param root: Root of tree
    :type root: list
    :param cnt: Current number of nodes in the tree
    :type cnt: int
    :return: Number of nodes in the tree
    :rtype: int
    """

    # Increase the count
    cnt += 1
    # Iterate over the children
    for child in get_children(root):
        # Recursively count the child nodes
        cnt = get_number_of_nodes(child, cnt)

    return cnt


def get_node_at_index(root, idx):
    """
    Return the node in the tree at a given index. The index is
    according to a depth-first left-to-right ordering.

    :param root: Root of tree
    :type root: list
    :param idx: Index of node to find
    :type idx: int
    :return: Node at the given index (based on depth-first left-to-right
      indexing)
    :rtype: list
    """

    # Stack of unvisited nodes
    unvisited_nodes = [root]
    # Initial node is the same as the root
    node = root
    # Set the current index
    cnt = 0
    # Iterate over the tree until the index is reached
    while cnt <= idx and unvisited_nodes:
        # Take an unvisited node from the stack
        node = unvisited_nodes.pop()
        # Add the children of the node to the stack
        # Get the children
        children = get_children(node)
        # Reverse the children before appending them to the stack
        children.reverse()
        # Add children to the stack
        unvisited_nodes.extend(children)

        # Increase the current index
        cnt += 1

    return node


def get_max_tree_depth(root, depth, max_tree_depth):
    """
    Return the max depth of the tree. Recursively traverse the tree

    :param root: Root of the tree
    :type root: list
    :param depth: Current tree depth
    :type depth: int
    :param max_tree_depth: Maximum depth of the tree
    :type max_tree_depth: int
    :return: Maximum depth of the tree
    :rtype: int
    """

    # Update the max depth if the current depth is greater
    if max_tree_depth < depth:
        max_tree_depth = depth

    # Traverse the children of the root node
    for child in get_children(root):
        # Increase the depth
        depth += 1
        # Recursively get the depth of the child node
        max_tree_depth = get_max_tree_depth(child, depth, max_tree_depth)
        # Decrease the depth
        depth -= 1

    assert depth <= max_tree_depth

    return max_tree_depth


def get_depth_at_index(node, idx, node_idx, depth, idx_depth=0):
    """
    Return the depth of a node based on the index. The index is based on
    depth-first left-to-right traversal.

    TODO implement breakout

    :param node: Current node
    :type node: list
    :param idx: Current index
    :type idx: int
    :param node_idx: Index of the node which depth we are searching for
    :type node_idx: int
    :param depth: Current depth
    :type depth: int
    :param idx_depth: Depth of the node at the given index
    :type idx_depth: int
    :return: Current index and depth of the node at the given index
    :rtype: tuple
    """
    # Assign the current depth when the current index matches the given index
    if node_idx == idx:
        idx_depth = depth

    idx += 1
    # Iterate over the children
    for child in get_children(node):
        # Increase the depth
        depth += 1
        # Recursively check the child depth and node index
        idx_depth, idx = get_depth_at_index(child, idx, node_idx, depth,
                                            idx_depth)
        # Decrease the depth
        depth -= 1

    return idx_depth, idx


def replace_subtree(new_subtree, old_subtree):
    """
    Replace a subtree.

    :param new_subtree: The new subtree
    :type new_subtree: list
    :param old_subtree: The old subtree
    :type old_subtree: list
    """

    # Delete the nodes of the old subtree
    del old_subtree[:]
    for node in new_subtree:
        # Insert the nodes in the new subtree
        old_subtree.append(copy.deepcopy(node))


def get_random_symbol(depth, max_depth, symbols, full=False):
    """
    Return a randomly chosen symbol. The depth determines if a terminal
    must be chosen. If `full` is specified a function will be chosen
    until the max depth. The symbol is picked with a uniform probability.

    :param depth: Current depth
    :type depth: int
    :param max_depth: Max depth determines if a function symbol can be
      chosen. I.e. a symbol with arity greater than zero
    :type max_depth: int
    :param symbols: The possible symbols.
    :type symbols: dict
    :param full: True if function symbols should be drawn until max depth
    :returns: A random symbol
    :rtype: str
    """

    # Pick a terminal if max depth has been reached
    if depth >= (max_depth - 1):
        # Pick a random terminal
        symbol = random.choice(symbols["terminals"])
    else:
        # Can it be a terminal before the max depth is reached
        # then there is 50% chance that it is a terminal
        if not full and bool(random.getrandbits(1)):
            # Pick a random terminal
            symbol = random.choice(symbols["terminals"])
        else:
            # Pick a random function
            symbol = random.choice(symbols["functions"])

    # Return the picked symbol
    return symbol


def sort_population(individuals):
    """
    Return a list sorted on the fitness value of the individuals in
    the population. Descending order.

    :param individuals: The population of individuals
    :type individuals: list
    :return: The population of individuals sorted by fitness in descending order
    :rtype: list
    """

    # Sort the individual elements on the fitness
    # Reverse for descending order
    individuals = sorted(individuals, key=lambda x: x['fitness'], reverse=True)

    return individuals


def evaluate_individual(individual, fitness_cases, targets):
    """
    Evaluate fitness based on fitness cases and target values. Fitness
    cases are a set of exemplars (input and output points) by
    comparing the error between the output of an individual(symbolic
    expression) and the target values.
    Evaluates and sets the fitness in an individual. Fitness is the
    negative mean square error(MSE).

    :param individual: Individual solution to evaluate
    :type individual: dict
    :param fitness_cases: Input for the evaluation
    :type fitness_cases: list
    :param targets: Output corresponding to the input
    :type targets: list
    """

    # Initial fitness value
    fitness = 0.0
    # Calculate the error between the output of the individual solution and
    # the target for each input
    for case, target in zip(fitness_cases, targets):
        # Get output from evaluation function
        output = evaluate(individual["genome"], case)
        # Get the squared error
        error = output - target
        fitness += error * error

    assert fitness >= 0
    # Get the mean fitness and assign it to the individual
    individual["fitness"] = -fitness / float(len(targets))

    assert individual['fitness'] <= 0


def evaluate(node, case):
    """
    Evaluate a node recursively. The node's symbol string is evaluated.

    :param node: Evaluated node
    :type node: list
    :param case: Current fitness case
    :type case: list
    :returns: Value of the evaluation
    :rtype: float
    """
    symbol = node[0]
    symbol = symbol.strip()

    # Identify the node symbol
    if symbol == "+":
        # Add the values of the node's children
        return evaluate(node[1], case) + evaluate(node[2], case)

    elif symbol == "-":
        # Subtract the values of the node's children
        return evaluate(node[1], case) - evaluate(node[2], case)

    elif symbol == "*":
        # Multiply the values of the node's children
        return evaluate(node[1], case) * evaluate(node[2], case)

    elif symbol == "/":
        # Divide the value's of the nodes children. Too low values of the
        # denominator returns the numerator
        numerator = evaluate(node[1], case)
        denominator = evaluate(node[2], case)
        if abs(denominator) < 0.00001:
            denominator = 1

        return numerator / denominator

    elif symbol.startswith("x"):
        # Get the variable value
        return case[int(symbol[1:])]
    else:
        # The symbol is a constant
        return float(symbol)


def initialize_population(param):
    """
    Ramped half-half initialization. The individuals in the
    population are initialized using the grow or the full method for
    each depth value (ramped) up to max_depth.

    :param param: parameters for pony gp
    :type param: dict
    :returns: List of individuals
    :rtype: list
    """

    individuals = []
    for i in range(param["population_size"]):
        # Pick full or grow method
        full = bool(random.getrandbits(1))
        # Ramp the depth
        max_depth = (i % param["max_depth"]) + 1
        # Create root node
        symbol = get_random_symbol(0, max_depth, param["symbols"])
        tree = [symbol]
        # Grow the tree if the root is a function symbol
        if max_depth > 0 and symbol in param["symbols"]["functions"]:
            grow(tree, 1, max_depth, full, param["symbols"])

            assert get_max_tree_depth(tree, 0, 0) < (max_depth + 1)

        # An individual is a dictionary
        individual = {"genome": tree, "fitness": DEFAULT_FITNESS}
        # Append the individual to the population
        individuals.append(individual)
        if param["verbose"]:
            print('Initial tree nr:{} nodes:{} max_depth:{} tree: {}'.format(
                i,
                get_number_of_nodes(tree, 0),
                get_max_tree_depth(tree, 0, 0), tree))

    return individuals


def coevaluate_fitness(population, adversaries, param, cache):
    """
    Evaluate and assign fitness to the population and the adversaries.

    Fitness is the average.

    :param population: Population to evaluate
    :type population: list
    :param adversaries: Population to evaluate
    :type adversaries: list
    :param param: parameters for pony gp
    :type param: dict
    :param cache: Cache for fitness evaluations
    :type cache: dict
    """
    population_fitnesses = collections.defaultdict(list)
    for adversary in adversaries:        
        exemplars = {"cases": [], "targets": []}
        adversary_fitness = []
        for case in param["fitness_cases"]:
            case_p = evaluate(adversary["genome"], case)
            case_p = [case_p - i for i in range(len(param["variables"]))]
            target = evaluate(param["system_function"], case_p)
            exemplars["cases"].append(case_p)
            exemplars["targets"].append(target)
            
        for individual in population:
            key = str(individual["genome"]) + str(exemplars)
            if key in cache.keys():
                individual["fitness"] = cache[key]
                print('\tC',individual["fitness"], individual["genome"])
            else:
                evaluate_individual(individual, exemplars["cases"],
                                    exemplars["targets"])
                cache[key] = individual["fitness"]
                print('\t',individual["fitness"], individual["genome"])
                
            adversary_fitness.append(individual["fitness"])
            population_fitnesses[str(individual["genome"])].append(individual["fitness"])

        adversary_fitness = -1.0 * (float(sum(adversary_fitness)) / float(len(population)))
        adversary["fitness"] = adversary_fitness
        print(adversary, exemplars)

    if param["verbose"]:
        print("CACHE SIZE: {} UNIQUE DEFENDER: {}/{} UNIQUE ATTACKER: {}/{}".format(
            len(cache), len(dict([(str(_["genome"]), None) for _ in population])),
            param["population_size"],
            len(dict([(str(_["genome"]), None) for _ in adversaries])),
            param["population_size"]))
        
    for individual_genome, fitnesses in population_fitnesses.items():
        fitness = float(sum(fitnesses)) / float(len(fitnesses))
        for individual in population:
            if individual_genome == str(individual["genome"]):
                individual["fitness"] = fitness
            
        
def search_loop(populations, param):
    """
    Return the best individual from the evolutionary search
    loop. Starting from the initial population.

    :param populations: Initial populations of individuals
    :type populations: dict
    :param param: parameters for pony gp
    :type param: dict
    :returns: Best individual
    :rtype: dict
    """

    # Evaluate fitness
    cache = {}
    best_ever = collections.OrderedDict()
    
    tic = time.time()
    coevaluate_fitness(populations["defender"], populations["attacker"], param, cache)
    # Print the stats of the population
    for name, population in populations.items():
        print_stats(0, population, time.time() - tic, param, name)
        # Set best solution
        sort_population(population)
        best_ever[name] = [population[0]]

    # Generation loop
    generation = 1
    while generation < param["generations"]:
        tic = time.time()
        new_populations = {}
        for name, population in populations.items():
            new_population = []
            # Selection
            parents = tournament_selection(population, param)

            # Crossover
            while len(new_population) < param["population_size"]:
                # Select parents
                _parents = random.sample(parents, 2)
                # Generate children by crossing over the parents
                children = subtree_crossover(_parents[0], _parents[1], param)
                # Append the children to the new population
                for child in children:
                    new_population.append(child)

            # Select population size individuals. Handles uneven population
            # sizes, since crossover returns 2 offspring
            new_population = new_population[:param["population_size"]]

            # Vary the population by mutation
            for i in range(len(new_population)):
                new_population[i] = subtree_mutation(new_population[i], param)

            new_populations[name] = new_population
            
        # Evaluate fitness
        coevaluate_fitness(new_populations["defender"],
                           new_populations["attacker"], param, cache)

        for name in populations.keys():
            # Replace population
            population = generational_replacement(new_populations[name],
                                                  populations[name],
                                                  param)

            # Set best solution
            sort_population(population)
            best_ever[name] = population[0]

            # Print the stats of the population
            print_stats(generation, population, time.time() - tic, param, name)

            populations[name] = population
            
        # Increase the generation counter
        generation += 1

    return best_ever


def print_stats(generation, individuals, duration, param, name=""):
    """
    Print the statistics for the generation and population.

    :param generation:generation number
    :type generation: int
    :param individuals: population to get statistics for
    :type individuals: list
    :param duration: duration of computation
    :type duration: float
    :param param: Parameters
    :type param: dict
    """

    def get_ave_and_std(values):
        """
        Return average and standard deviation.
        :param values: Values to calculate on
        :type values: list
        :returns: Average and Standard deviation of the input values
        :rtype: tuple
        """
        _ave = float(sum(values)) / len(values)
        _std = math.sqrt(
            float(sum((value - _ave)**2 for value in values)) / len(values))
        return _ave, _std

    # Make sure individuals are sorted
    individuals = sort_population(individuals)
    if param["verbose"]:
        print("{} POPULATION: {}".format(name, individuals))
    # Get the fitness values
    fitness_values = [i["fitness"] for i in individuals]
    # Get the number of nodes
    size_values = [get_number_of_nodes(i["genome"], 0) for i in individuals]
    # Get the max depth
    depth_values = [get_max_tree_depth(i["genome"], 0, 0) for i in individuals]
    # Get average and standard deviation of fitness
    ave_fit, std_fit = get_ave_and_std(fitness_values)
    # Get average and standard deviation of size
    ave_size, std_size = get_ave_and_std(size_values)
    # Get average and standard deviation of max depth
    ave_depth, std_depth = get_ave_and_std(depth_values)
    # Print the statistics
    print(
        "%s Generation:%d Duration: %.4f fit_ave:%.2f+-%.3f size_ave:%.2f+-%.3f "
        "depth_ave:%.2f+-%.3f max_size:%d max_depth:%d max_fit:%f "
        "best_solution:%s" %
        (name, generation, duration, ave_fit, std_fit, ave_size, std_size, ave_depth,
         std_depth, max(size_values), max(depth_values), max(fitness_values),
         individuals[0]))


def subtree_mutation(individual, param):
    """Subtree mutation. Pick a node and grow it.

        :param individual: Individual to mutate
        :type individual: dict
        :param param: parameters for pony gp
        :type param: dict
        :returns: Mutated individual
        :rtype: dict

    """
    
    # Copy the individual for mutation
    new_individual = {
        "genome": copy.deepcopy(individual["genome"]),
        "fitness": DEFAULT_FITNESS
    }
    # Check if mutation should be applied
    if random.random() < param["mutation_probability"]:
        # Pick node
        end_node_idx = get_number_of_nodes(new_individual["genome"], 0) - 1
        node_idx = random.randint(0, end_node_idx)
        node = get_node_at_index(new_individual["genome"], node_idx)
        old_node = node[:]
        # Clear children
        node_depth = get_depth_at_index(new_individual["genome"], 0, node_idx, 0)[0]
        new_symbol = get_random_symbol(node_depth, param["max_depth"] - node_depth, param["symbols"])
        new_subtree = [new_symbol]
        # Grow tree
        grow(new_subtree, node_depth, param["max_depth"], False, param["symbols"])
        replace_subtree(new_subtree, node)
        if param["verbose"]:
            print("MUTATED: {} to {}. subtree: {} to subtree: {} at idx {}".format(
                new_individual["genome"], individual["genome"], old_node, new_subtree, node_idx))
        
    return new_individual


def subtree_crossover(parent1, parent2, param):
    """
    Returns two individuals. The individuals are created by
    selecting two random nodes from the parents and swapping the
    subtrees.

    :param parent1: Parent one to crossover
    :type parent1: dict
    :param parent2: Parent two to crossover
    :type parent2: dict
    :param param: parameters for pony gp
    :type param: dict
    :return: Children from the crossed over parents
    :rtype: tuple
    """
    # Copy the parents to make offsprings
    offsprings = ({
        "genome": copy.deepcopy(parent1["genome"]),
        "fitness": DEFAULT_FITNESS
    }, {
        "genome": copy.deepcopy(parent2["genome"]),
        "fitness": DEFAULT_FITNESS
    })

    # Check if offspring will be crossed over
    if random.random() < param["crossover_probability"]:
        xo_nodes = []
        node_depths = []
        for i, offspring in enumerate(offsprings):
            # Pick a crossover point
            end_node_idx = get_number_of_nodes(offsprings[i]["genome"], 0) - 1
            node_idx = random.randint(0, end_node_idx)
            # Find the subtree at the crossover point
            xo_nodes.append(
                get_node_at_index(offsprings[i]["genome"], node_idx))
            xo_point_depth = get_max_tree_depth(xo_nodes[-1], 0, 0)
            offspring_depth = get_max_tree_depth(offspring["genome"], 0, 0)
            node_depths.append((xo_point_depth, offspring_depth))

        # Make sure that the offspring is deep enough
        if (node_depths[0][1] + node_depths[1][0]) >= param["max_depth"] or \
                        (node_depths[1][1] + node_depths[0][0]) >= param["max_depth"]:
            if param["verbose"]:
                print("Crossover is too deep: {} or {} is > {}".format((node_depths[0][
                    1] + node_depths[1][0]), (node_depths[1][1] + node_depths[
                        0][0]), param["max_depth"]))
            return offsprings

        if param["verbose"]:
            print("CROSSOVER {} to {}".format(xo_nodes[0], xo_nodes[1]))

        # Swap the nodes
        tmp_offspring_1_node = copy.deepcopy(xo_nodes[1])
        # Copy the children from the subtree of the first offspring
        # to the chosen node of the second offspring
        replace_subtree(xo_nodes[0], xo_nodes[1])
        # Copy the children from the subtree of the second offspring
        # to the chosen node of the first offspring
        replace_subtree(tmp_offspring_1_node, xo_nodes[0])

        for offspring in offsprings:
            assert get_max_tree_depth(offspring["genome"], 0,
                                      0) <= param["max_depth"]

    # Return the offsprings
    return offsprings


def tournament_selection(population, param):
    """
    Return individuals from a population by drawing
    `tournament_size` competitors randomly and selecting the best
    of the competitors. `population_size` number of tournaments are
    held.

    :param population: Population to select from
    :type population: list
    :param param: parameters for pony gp
    :type param: dict
    :returns: selected individuals
    :rtype: list
    """

    # Iterate until there are enough tournament winners selected
    winners = []
    while len(winners) < param["population_size"]:
        # Randomly select tournament size individual solutions
        # from the population.
        competitors = random.sample(population, param["tournament_size"])
        # Rank the selected solutions
        competitors = sort_population(competitors)
        # Append the best solution to the winners
        winners.append(competitors[0])

    return winners


def generational_replacement(new_population, old_population, param):
    """
    Return new a population. The `elite_size` best old_population
    are appended to the new population. They are kept in the new
    population if they are better than the worst.
    population if they are better than the worst.

    :param new_population: the new population
    :type new_population: list
    :param old_population: the old population
    :type old_population: list
    :param param: parameters for pony gp
    :type param: dict
    :returns: the new population with the best from the old population
    :rtype: list
    """

    # Sort the population
    old_population = sort_population(old_population)
    # Append a copy of the best solutions of the old population to
    # the new population. ELITE_SIZE are taken
    for ind in old_population[:param["elite_size"]]:
        new_population.append(copy.deepcopy(ind))

    # Sort the new population
    new_population = sort_population(new_population)

    # Set the new population size
    return new_population[:param["population_size"]]


def run(param):
    """
    Return the best solution. Create an initial
    population. Perform an evolutionary search.

    :param param: parameters for pony gp
    :type param: dict
    :returns: Best solution
    :rtype: dict
    """

    # Create population
    populations = {}
    for population_name in param["populations"]:
        populations[population_name] = initialize_population(param)

    # Start evolutionary search
    best_ever = search_loop(populations, param)

    return best_ever


def get_symbols(arities):
    """Return a symbol dictionary. Helper method to keep the code clean. 

    The nodes in a GP tree consists of different symbols. The symbols
    are either functions (internal nodes with arity > 0) or terminals
    (leaf nodes with arity = 0) The symbols are represented as a
    dictionary with the keys: 
      - *arities* -- A dictionary where a key
        is a symbol and the value is the arity 
      - *terminals* -- A list of
        strings(symbols) with arity 0 
      - *functions* -- A list of
        strings(symbols) with arity > 0

    :param arities: A dictionary where a key
    is a symbol and the value is the arity
    :type arities: dict
    :return: Symbols used for GP individuals
    :rtype: dict

    """

    # List of terminal symbols
    terminals = []
    # List of functions
    functions = []
    # Append symbols to terminals or functions by looping over the arities items
    for key, value in arities.items():
        # A symbol with arity 0 is a terminal
        if value == 0:
            # Append the symbols to the terminals list
            terminals.append(key)
        else:
            # Append the symbols to the functions list
            functions.append(key)

    return {"arities": arities, "terminals": terminals, "functions": functions}


def get_arities(param, outputs=1):
    """Assign values to arities dictionary. Variables are taken from
    fitness case file header. Constants are read from config file.

    :param param: The dictionary of parameters
    :type param: dictionary
    :param outputs: The number of output columns from the input
    :type outputs: int
    :return: The arities dictionary
    :rtype: dictionary

    """

    assert outputs > 0

    arities = param['arities']
    # Input variables are all
    variables = param["variables"]
    for variable in variables:
        arities[variable.strip()] = 0

    # Add constant values
    constants = param['constants']
    if constants:
        for constant in constants:
            arities[str(constant)] = 0

    return arities


def get_test_and_train_data(fitness_cases_file, test_train_split):
    """Return test and train data. Random selection or exemplars(ros)
    from file containing data.

    :param fitness_cases_file: CSV file with a header.
    :type fitness_cases_file: str
    :param test_train_split: Percentage of exemplar data used for training
    :type test_train_split: float
    :return: Test and train data. Both cases and targets
    :rtype: tuple
    """

    exemplars, targets = parse_exemplars(fitness_cases_file)
    split_idx = int(math.floor(len(exemplars) * test_train_split))
    # Randomize
    idx = list(range(0, len(exemplars)))
    random.shuffle(idx)
    training_cases = []
    training_targets = []
    test_cases = []
    test_targets = []
    for i in idx[:split_idx]:
        training_cases.append(exemplars[i])
        training_targets.append(targets[i])

    for i in idx[split_idx:]:
        test_cases.append(exemplars[i])
        test_targets.append(targets[i])

    return ({
        "fitness_cases": test_cases,
        "targets": test_targets
    }, {
        "fitness_cases": training_cases,
        "targets": training_targets
    })


def parse_arguments():
    """
    Returns a dictionary of the default parameters, or the ones set by
    commandline arguments

    :return: parameters for the GP run
    :rtype: dict
    """
    # Command line arguments
    parser = argparse.ArgumentParser()
    # Population size
    parser.add_argument(
        "-p",
        "--population_size",
        type=int,
        dest="population_size",
        help="Population size is the number of individual "
        "solutions")
    # Size of an individual
    parser.add_argument(
        "-m",
        "--max_depth",
        type=int,
        dest="max_depth",
        help="Max depth of tree. Partly determines the search "
        "space of the solutions")
    # Number of elites, i.e. the top solution from the old population
    # transferred to the new population
    parser.add_argument(
        "-e",
        "--elite_size",
        type=int,
        dest="elite_size",
        help="Elite size is the number of best individual "
        "solutions that are preserved between generations")
    # Generations is the number of times the EA will iterate the search loop
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        dest="generations",
        help="Number of generations. The number of iterations "
        "of the search loop.")
    # Tournament size
    parser.add_argument(
        "--ts",
        "--tournament_size",
        type=int,
        dest="tournament_size",
        help="Tournament size. The number of individual "
        "solutions that are compared when determining "
        "which solutions are inserted into the next "
        "generation(iteration) of the search loop")
    # Random seed.
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        dest="seed",
        help="Random seed. For replication of runs of the EA. "
        "The search is stochastic and and replication of "
        "the results are guaranteed the random seed")
    # Probability of crossover
    parser.add_argument(
        "--cp",
        "--crossover_probability",
        type=float,
        dest="crossover_probability",
        help="Crossover probability, [0.0,1.0]. The probability "
        "of two individual solutions to be varied by the "
        "crossover operator")
    # Probability of mutation
    parser.add_argument(
        "--mp",
        "--mutation_probability",
        type=float,
        dest="mutation_probability",
        help="Mutation probability, [0.0, 1.0]. The probability "
        "of an individual solutions to be varied by the "
        "mutation operator")
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file in Python INI format. Overridden by CLI-arguments.")
    # Verbose
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action='store_const',
        const=True,
        help="Verbose printing")

    # Parse the command line arguments
    args = parser.parse_args()
    param = parse_config_file(args, parser)

    # Override config file values with CLI-args
    _args = vars(args)
    for key, value in _args.items():
        if value is not None or key is 'verbose':
            param[key] = value

    return param


def parse_config_file(args, parser):
    """Parse configuration file.

    :param args: CLI arguments
    :type args: argparse.Namespace
    :param parser: CLI parser
    :type parser: argparse.ArgumentParser
    :return: Parameters
    :rtype: dict
    """
    param = {}
    config_parser = configparser.ConfigParser()
    config_parser.read(args.config)
    # Parse function(non-terminal) symbols and arity
    param['arities'] = {}
    for k, v in config_parser['arities'].items():
        param['arities'][k] = int(v)

    # Parse constants
    param['constants'] = [float(_.strip()) for _ in config_parser['constants']['values'].split(',')]

    # Parse populations
    param['populations'] = [_.strip() for _ in config_parser['Coevolution']['populations'].split(',')]

    # Get System function
    param["system_function"] = eval(config_parser["Fitness function"]["system_function"])

    # Get fitness cases
    param["fitness_cases"] = eval(config_parser["Fitness function"]["fitness_cases"])

    # Get variables
    param["variables"] = eval(config_parser["Fitness function"]["variables"])
    
    # Get the type of the search parameter by using the argument parser
    tmp = ['--config', 'True']
    for key, value in config_parser['Search parameters'].items():
        tmp.append('--{}'.format(key))
        tmp.append(value)
    tmp = parser.parse_args(tmp)
    for key, value in vars(tmp).items():
        param[key] = value

    return param


def main():
    """Search. Evaluate best solution on out-of-sample data"""

    # Set arguments
    param = parse_arguments()
    seed = param["seed"]
    # Set random seed if not 0 is passed in as the seed
    if seed != 0:
        random.seed(seed)
    # Get the symbols
    arities = get_arities(param)
    symbols = get_symbols(arities)
    # Get the namespace dictionary
    param["symbols"] = symbols

    # Print GP settings
    print("GP settings:\n {}".format(param))

    # Run
    best_ever = run(param)
    for key, value in best_ever.items():
        print("{} End solutions: {}".format(key, value))


if __name__ == '__main__':
    main()
