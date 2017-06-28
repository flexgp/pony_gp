import yaml
import csv

with open("user_parameters.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#creates the param dictionary
param=(cfg['param'])

#creates the arities dictionary
def update_arities():
    arities=(cfg['arities'])
    with open("fitness_cases.csv", "r") as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        # Read the header
        headers = reader.__next__()
        for val in headers[:-1]:
            arities[val] = 0
        range_of_numbers = (cfg['range_of_numbers'])
        constants = range(0, (range_of_numbers+1))
        for constant in constants:
            arities[str(constant)] =0
    return arities
