from experiment import ExperimentCollection, Experiment
import argparse

default_experiment_name = None
default_collection_name = 'prototyping'

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--collection_name', type=str, default=default_collection_name)
#     parser.add_argument('--experiment_name', type=str, default=default_experiment_name)
#
#     args = parser.parse_args()
#
#     if args.collection_name is not None:
#         xp = ExperimentCollection(args.collection_name)
#     elif args.experiment_name is not None:
#         xp = Experiment(args.experiment_name)
#
#
#     xp.load()
#     xp.analyze()



experiment_name = None
collection_name = 'prototyping'

if collection_name is not None:
    xp = ExperimentCollection(collection_name)
elif experiment_name is not None:
    xp = Experiment(experiment_name)


xp.load()
xp.analyze()
