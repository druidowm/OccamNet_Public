import json
import argparse
from experiment import ExperimentCollection, Experiment

experiments_folder = "./experiments/"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', type=str, default='basic_collection')
    args = parser.parse_args()

    experiment_collection = ExperimentCollection(args.collection_name)
    with open(experiments_folder + args.collection_name + '.json', 'r') as f:
        parameters = json.load(f)
        for experiment_params in parameters['collection']:
            experiment_collection.push(Experiment(**experiment_params))

    experiment_collection.run()
