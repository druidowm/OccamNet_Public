import json
import argparse
from experiment import ExperimentCollection, Experiment
import torch
import numpy as np

experiments_folder = "./experiments/"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', type=str, default='pattern_recognition')
    parser.add_argument('--name', type=str, default='pattern_recognition')
    parser.add_argument('--skip_connections', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--number_of_outputs', type=int, default=2)
    parser.add_argument('--number_of_inputs', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--last_layer_temperature', type=float, default=10)
    parser.add_argument('--sampling_size', type=int, default=150)
    parser.add_argument('--truncation_parameter', type=int, default=5)
    parser.add_argument('--variances', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # sets the seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    experiment_collection = ExperimentCollection(args.collection_name)
    with open(experiments_folder + args.collection_name + '.json', 'r') as f:
        parameters = json.load(f)
        for experiment_params in parameters['collection']:
            var_dict = vars(args)
            for d in var_dict:
                if d == 'collection_name':
                    continue
                elif d == 'seed':
                    continue
                experiment_params[d] = var_dict[d]  # overriding

            experiment_collection.push(Experiment(**experiment_params))

    experiment_collection.run()
