import os

###############################
# HERE WE SPECIFY THE CORE
###############################

variables = {
    'collection_name': 'pattern_recognition',
    'name': 'default',
    'skip_connections': False,
    'imagenet': False,
    'learning_rate': 0.05,
    'depth': 2,
    'number_of_outputs': 3,
    'number_of_inputs': 784,
    'temperature': 1,
    'last_layer_temperature': 10,
    'sampling_size': 150,
    'truncation_parameter': 10,
    'variances': 0.01,
    'epochs': 5000,
    'batch_size': 1000,
    'finetune': False,
    'resnet': False
}

###############################
# HERE WE SPECIFY THE VARIABLES
###############################

NUM_GPU = 16
# main_path = 'scripts'
main_path = ""

# learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
# depth = [2, 3, 4, 5, 6, 7, 8, 9]
# variances = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
seeds = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
# skip_connections = [True, False]
# assert NUM_GPU == len(variances) * len(skip_connections)
# assert NUM_GPU == len(learning_rate) * len(skip_connections)
# assert NUM_GPU == len(depth) * len(skip_connections)

idx = 0
for i in seeds:
# for i in learning_rate:
# for i in depth:
# for i in variances:
#     for j in skip_connections:
        idx += 1
        # variables['name'] = f"Exp1_Ind{idx}_learningrate{i}_skip{j}"
        # variables['name'] = f"Exp2_10classes_Ind{idx}_depth{i}_skip{j}"
        # variables['name'] = f"Exp3_3classes_Ind{idx}_learningrate{i}_skip{j}"
        # variables['name'] = f"Exp4_imagenet_Ind{idx}_learningrate{i}_skip{j}"
        # variables['name'] = f"Exp5_imagenet_crazy_Ind{idx}_learningrate{i}_skip{j}"
        # variables['name'] = f"Exp6_imagenet_cresnet_Ind{idx}_seed_{i}"
        # variables['name'] = f"Exp7_imagenet_cresnet_Ind{idx}_seed_{i}"
        # variables['name'] = f"Exp9_imagenet_finetune_Ind{idx}_seed_{i}"
        # variables['name'] = f"Exp10_imagenet_finetune_resnet_Ind{idx}_seed_{i}"
        # variables['name'] = f"Exp11_mnist_binary_Ind{idx}_seed_{i}"
        variables['name'] = f"Exp12_mnist_trinary_Ind{idx}_seed_{i}"
        variables['seed'] = i
        # variables['variances'] = i
        # variables['learning_rate'] = i
        # variables['depth'] = i
        # variables['skip_connections'] = j
        # if variables['skip_connections']:
        #     variables['learning_rate'] = 0.0005
        # else:
        #     variables['learning_rate'] = 0.05

        # CREATE THE FILE
        job_name = variables['name']
        command = f"#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=20\n#SBATCH -o logs/{job_name}.out\n#SBATCH --job-name={job_name}\n\n"
        command += 'python -u run_experiment.py '
        for d in variables:
            if d == 'skip_connections':
                if variables[d]:
                    command += '--' + d + ' '
            elif d == 'imagenet':
                if variables['imagenet']:
                    command += '--' + d + ' '
            elif d == 'finetune':
                if variables['finetune']:
                    command += '--' + d + ' '
            elif d == 'resnet':
                if variables['resnet']:
                    command += '--' + d + ' '
            else:
                command += '--' + d + ' ' + str(variables[d]) + ' '
        command += '\n'
        filename = os.path.join(main_path, variables['name'] + '.sh')
        with open(filename, 'w') as file:
            file.write(command)
            file.close()
        print(f"Finished writing file {filename}")
