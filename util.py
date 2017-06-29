import yaml
import csv


def get_param_and_arities():
    """
       Load values from user_parameters file. Return param and arities dictionaries.
       :return: The param dictionary
       :rtype: dictionary
       :return: The arities dictionary
       :rtype: dictionary
       """
    with open("user_parameters.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return get_param(cfg), get_arities(cfg)


def get_param(config):
    """
    Assign values to param dictionary. Return param dictionary.
    :param config: The dictionary that loads data from user_parameters.yml
    :type config: dictionary
    :return: The param dictionary
    :rtype: dictionary
       Assigns values to the arities dictionary, and then returns the arities dictionary.
    """

    param = (config['param'])

    return param


def get_arities(config):
    """
    Assign values to arities dictionary. Return arities dictionary.
    :param config: The dictionary that loads data from user_parameters.yml
    :type config: dictionary
    :return: The arities dictionary
    :rtype: dictionary
    Assigns values to the arities dictionary, and then returns the arities dictionary.
    """
    arities = (config['arities'])
    with open("fitness_cases.csv", "r") as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        # Read the header in order to define the input arities as 0
        headers = reader.__next__()

    for val in headers[:-1]:
        arities[val] = 0
    range_of_numbers = (config['range_of_numbers'])
    constants = range(0, (range_of_numbers + 1))
    for constant in constants:
        arities[str(constant)] = 0

    return arities
