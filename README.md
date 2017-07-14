`pony_gp` is an implementation of Genetic Programming(GP), see e.g.
<http://geneticprogramming.com>. The purpose of `pony_gp` is to describe how
the GP algorithm works. The intended use is for teaching. The aim is to allow
the developer to quickly start using and developing. The design is supposed
to be simple, self contained and use core python libraries.

# Run

Find a equation for the input given an output.

```python python pony_gp.py --config=configs.ini ``` 

Example output:
```
Reading: fitness_cases.csv headers: ['# x0', 'x1', 'y'] exemplars:121
GP settings:
{'arities': {'+': 2, '*': 2, '/': 2, '-': 2, 'x0': 0, 'x1': 0, '0.0': 0, '1.0': 0}, 'constants': [0.0, 1.0], 'population_size': 4, 'max_depth': 5, 'elite_size': 2, 'generations': 2, 'tournament_size': 3, 'seed': 0, 'crossover_probability': 0.8, 'mutation_probability': 0.2, 'fitness_cases': [[-3.0, 4.0], [-2.0, 3.0], [-4.0, 3.0], [-5.0, -3.0], [5.0, -3.0], [0.0, -1.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, -3.0], [-4.0, -2.0], [-1.0, -2.0], [5.0, 1.0], [-5.0, -1.0], [-1.0, 3.0], [4.0, 5.0], [-2.0, 1.0], [3.0, 1.0], [-3.0, 0.0], [-1.0, -4.0], [0.0, 3.0], [3.0, -3.0], [0.0, 1.0], [5.0, -2.0], [2.0, 1.0], [1.0, 3.0], [4.0, 4.0], [0.0, -4.0], [-1.0, 1.0], [-4.0, 4.0], [-5.0, 4.0], [-2.0, 0.0], [-4.0, 1.0], [-3.0, 3.0], [2.0, 5.0], [-2.0, -4.0], [2.0, -2.0], [0.0, 4.0], [0.0, -5.0], [1.0, 4.0], [5.0, 0.0], [-5.0, 5.0], [4.0, 3.0], [5.0, 2.0], [3.0, 2.0], [2.0, -1.0], [-5.0, 2.0], [-3.0, -2.0], [2.0, 2.0], [4.0, -5.0], [3.0, 4.0], [-1.0, 2.0], [-4.0, -5.0], [-5.0, -4.0], [3.0, 0.0], [-2.0, -5.0], [-3.0, -1.0], [5.0, 5.0], [-2.0, 2.0], [4.0, 1.0], [-5.0, -5.0], [4.0, -2.0], [-3.0, -4.0], [-4.0, -1.0], [1.0, 2.0], [-3.0, 2.0], [-5.0, 3.0], [4.0, 0.0], [3.0, -1.0], [-3.0, 1.0], [-3.0, 5.0], [1.0, -4.0], [2.0, 3.0], [2.0, -3.0], [1.0, -3.0], [5.0, -4.0], [1.0, 5.0], [-2.0, 4.0], [5.0, -5.0], [-5.0, 0.0], [2.0, -5.0], [1.0, -2.0], [1.0, 1.0], [4.0, -4.0], [-1.0, -5.0]], 'test_train_split': 0.7, 'config': 'configs.ini', 'verbose': None, 'symbols': {'arities': {'+': 2, '*': 2, '/': 2, '-': 2, 'x0': 0, 'x1': 0, '0.0': 0, '1.0': 0}, 'terminals': ['x0', 'x1', '0.0', '1.0'], 'functions': ['+', '*', '/', '-']}, 'targets': [25.0, 13.0, 25.0, 34.0, 34.0, 1.0, 4.0, 1.0, 13.0, 20.0, 5.0, 26.0, 26.0, 10.0, 41.0, 5.0, 10.0, 9.0, 17.0, 9.0, 18.0, 1.0, 29.0, 5.0, 10.0, 32.0, 16.0, 2.0, 32.0, 41.0, 4.0, 17.0, 18.0, 29.0, 20.0, 8.0, 16.0, 25.0, 17.0, 25.0, 50.0, 25.0, 29.0, 13.0, 5.0, 29.0, 13.0, 8.0, 41.0, 25.0, 5.0, 41.0, 41.0, 9.0, 29.0, 10.0, 50.0, 8.0, 17.0, 50.0, 20.0, 25.0, 17.0, 5.0, 13.0, 34.0, 16.0, 10.0, 10.0, 34.0, 17.0, 13.0, 13.0, 10.0, 41.0, 26.0, 20.0, 50.0, 25.0, 29.0, 5.0, 2.0, 32.0, 26.0]}
Generation:0 Duration: 0.0016 fit_ave:-572.76+-25.137 size_ave:2.00+-1.000 depth_ave:0.50+-0.500 max_size:3 max_depth:1 max_fit:-530.166667 best_solution:{'genome': ['1.0'], 'fitness': -530.1666666666666}
Generation:1 Duration: 0.0035 fit_ave:-530.17+-0.000 size_ave:1.00+-0.000 depth_ave:0.00+-0.000 max_size:1 max_depth:0 max_fit:-530.166667 best_solution:{'genome': ['1.0'], 'fitness': -530.1666666666666}
Best solution on train data:{'genome': ['1.0'], 'fitness': -530.1666666666666}
Best solution on test data:{'genome': ['1.0'], 'fitness': -487.1081081081081}
```

If you wish to,
change the paramaters from the 'configs.ini' file to your desired
paramaters or allow it to remain at its default values.

The input with their respective output is in the file `fitness_case.csv`. The
exemplars are generated from `y = x0^2 + x1^2` from range `[-5,5]`

## Requirements

Python 3

## Usage

```
usage: pony_gp.py [-h] [-p POPULATION_SIZE] [-m MAX_DEPTH] [-e ELITE_SIZE]
                  [-g GENERATIONS] [--ts TOURNAMENT_SIZE] [-s SEED]
                  [--cp CROSSOVER_PROBABILITY] [--mp MUTATION_PROBABILITY]
                  [--fc FITNESS_CASES] [--tts TEST_TRAIN_SPLIT] --config
                  CONFIG [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  -p POPULATION_SIZE, --population_size POPULATION_SIZE
                        Population size is the number of individual solutions
  -m MAX_DEPTH, --max_depth MAX_DEPTH
                        Max depth of tree. Partly determines the search space
                        of the solutions
  -e ELITE_SIZE, --elite_size ELITE_SIZE
                        Elite size is the number of best individual solutions
                        that are preserved between generations
  -g GENERATIONS, --generations GENERATIONS
                        Number of generations. The number of iterations of the
                        search loop.
  --ts TOURNAMENT_SIZE, --tournament_size TOURNAMENT_SIZE
                        Tournament size. The number of individual solutions
                        that are compared when determining which solutions are
                        inserted into the next generation(iteration) of the
                        search loop
  -s SEED, --seed SEED  Random seed. For replication of runs of the EA. The
                        search is stochastic and and replication of the
                        results are guaranteed the random seed
  --cp CROSSOVER_PROBABILITY, --crossover_probability CROSSOVER_PROBABILITY
                        Crossover probability, [0.0,1.0]. The probability of
                        two individual solutions to be varied by the crossover
                        operator
  --mp MUTATION_PROBABILITY, --mutation_probability MUTATION_PROBABILITY
                        Mutation probability, [0.0, 1.0]. The probability of
                        an individual solutions to be varied by the mutation
                        operator
  --fc FITNESS_CASES, --fitness_cases FITNESS_CASES
                        Fitness cases filename. The exemplars of input and the
                        corresponding out put used to train and test
                        individual solutions
  --tts TEST_TRAIN_SPLIT, --test_train_split TEST_TRAIN_SPLIT
                        Test-train data split, [0.0,1.0]. The ratio of fitness
                        cases used for training individual solutions
  --config CONFIG       Config file in Python INI format. Overridden by CLI-
                        arguments.
  --verbose, -v         Verbose printing
```

## Output
Keep running until fitness for train data and test data reaches approximately 0.

### Statistics on Data
Reading: csv file containing input and output data for program to execute
         headers: [input(s), output] exemplars: amount of input and output points

### Individual Statistics

`Initial individual nr`:individual number nodes: amount of nodes or
different symbols in the individual, `max_depth`: max depth of
individual(refer to usage): individual generated

### Generation Statistics
`Generation`:generation number, `duration`:evaluation time, `fit_ave`:average fitness of the generation, `size_ave`:average number of nodes in the genearation amongst all solutions, `depth_ave`:average max_tree depth, max_size`: maximum number of nodes, `max_depth`: maximum depth, `max_fit`: maximum fitnessm `best_solution`:{`'genome'`: individual formula/tree, `'fitness'`: fitness of genome}

### Best Solution Statistics
```
Best solution on train data:{'genome': individual formula/tree, 'fitness': fitness of genome}
Best solution on test data:{'genome':individual formula/tree, 'fitness':fitness of genome}
```
