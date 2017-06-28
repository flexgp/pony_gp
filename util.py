import yaml
import csv

with open("user_parameters.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#creates the param dictionary
def create_param():
    param=(cfg['param'])

    return param

#creates the arities dictionary
def create_arities(config):
    arities = (config['arities'])
    with open("fitness_cases.csv", "r") as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        # Read the header in order to define the input arities as 0
        headers = reader.__next__()

    for val in headers[:-1]:
        arities[val] = 0
    range_of_numbers = (cfg['range_of_numbers'])
    constants = range(0, (range_of_numbers + 1))
    for constant in constants:
        arities[str(constant)] = 0

    return arities

def get_config():
    with open("user_parameters.yml", 'r') as ymlfile:
        return yaml.load(ymlfile)