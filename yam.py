#All changes made by Mahek

import yaml
import csv

with open("configuration.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

generation_size=(cfg['generation size'])
population_size=(cfg['population size'])
max_depth=(cfg['max depth'])
elite_size=(cfg['elite size'])
tournament_size=(cfg['tournament size'])
crossover_probability=(cfg['crossover probability'])
mutation_probability=(cfg['mutation probability'])
fitness_case=(cfg['fitness case'])

rangeofnumbers= (cfg['rangeofnumbers'])
arities=(cfg['arities'])
with open("fitness_cases.csv", "r") as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    # Read the header
    headers = reader.__next__()
    for val in headers[:-1]:
        arities[val] = 0
    constants = range(0, (rangeofnumbers+1))
    for constant in constants:
        arities[str(constant)] =0
