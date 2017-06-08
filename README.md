`pony_gp` is an implementation of Genetic Programming(GP), see e.g. 
<http://geneticprogramming.com>. The purpose of `pony_gp` is to describe how 
the GP algorithm works. The intended use is for teaching. The aim is to allow
the developer to quickly start using and developing. The design is supposed 
to be simple, self contained and use core python libraries. 

# Run

Find a equation for the input given an output.

```python
python pony_gp.py
```

The input with their respective output is in the file `fitness_case.csv`. The 
exemplars are generated from `y = x0^2 + x1^2` from range `[-5,5]`

## Requirements

Python 3

## Usage

```
Usage: pony_gp.py [options]

Arguments:
  -h, --help            show this help message and exit
  -p POPULATION_SIZE, --population_size=POPULATION_SIZE
                        Population size is the number of individual solutions
  -m MAX_DEPTH, --max_depth=MAX_DEPTH
                        Max depth of tree. Partly determines the search space
                        of the solutions
  -e ELITE_SIZE, --elite_size=ELITE_SIZE
                        Elite size is the number of best individual solutions
                        that are preserved between generations
  -g GENERATIONS, --generations=GENERATIONS
                        Number of generations. The number of iterations of the
                        search loop.
  --ts=TOURNAMENT_SIZE, --tournament_size=TOURNAMENT_SIZE
                        Tournament size. The number of individual solutions
                        that are compared when determining which solutions are
                        inserted into the next generation(iteration) of the
                        search loop
  -s SEED, --seed=SEED  Random seed. For replication of runs of the EA. The
                        search is stochastic and and replication of the
                        results are guaranteed the random seed
  --cp=CROSSOVER_PROBABILITY, --crossover_probability=CROSSOVER_PROBABILITY
                        Crossover probability, [0.0,1.0]. The probability of
                        two individual solutions to be varied by the crossover
                        operator
  --mp=MUTATION_PROBABILITY, --mutation_probability=MUTATION_PROBABILITY
                        Mutation probability, [0.0, 1.0]. The probability of
                        an individual solutions to be varied by the mutation
                        operator
  --fc=FITNESS_CASES, --fitness_cases=FITNESS_CASES
                        Fitness cases filename. The exemplars of input and the
                        corresponding out put used to train and test
                        individual solutions
  --tts=TEST_TRAIN_SPLIT, --test_train_split=TEST_TRAIN_SPLIT
                        Test-train data split, [0.0,1.0]. The ratio of fitness
                        cases used for trainging individual solutions
```             

## Output
```
#Statistics on Data
Reading: csv file containing input and output data for program to execute 
         headers: [input(s), output] exemplars: amount of input and output points   

#Refer to the above usage section to apprehend this line
#Statistics on the different paramaters
Namespace(crossover_probability= default crossover probability, elite_size= default elite_size, 
         fitness_cases= default fitness_cases, generations= default genarations, max_depth=default max_depth, 
         mutation_probability= default mutation_probability, population_size= default pupoulation_size,
         seed= default seed, test_train_split= default test_train_split, tournament_size= default tournament_size) 
         {'arities': {number of arguments each corresponding function takes}, 
         'terminals': [list of terminals(functions that have an arity of 0)], 
         'functions': [list of function sybols]}

#Individual/Tree Statistics
Initial tree nr:tree number nodes: amount of nodes or different symbols in the tree
max_depth: max depth of tree(refer to usage): tree generated

#Generation Statistics
Generation:generation number fit_ave:average fitness of the generation 
           size_ave:average number of nodes in the genearation amongst all data points 
           depth_ave:average max_tree depth max_size: maximum number of nodes 
           max_depth: maximum depth max_fit: maximum fitness 
           best_solution:{'genome': individual formula/tree, 'fitness': fitness of genome}

#Best Solution Statistics
Best solution on train data:{'genome': individual formula/tree, 'fitness': fitness of genome}
Best solution on test data:{'genome':individual formula/tree, 'fitness':fitness of genome}

``` 
Keep running until fitness for train data and test data reaches approximately 0.
