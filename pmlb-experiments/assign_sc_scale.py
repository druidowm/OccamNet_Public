import pickle
import os


def assign_jobs(i):
    # creates the executable script
    preamble = f'#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=20\n#SBATCH' \
               + f'-o logs/{i}.out\n#SBATCH --job-name={i}\n\n'
    with open(f'./scripts/{i}.sh', 'w') as file:
        file.write(preamble)
        file.write(f'python -u run_sc_scale.py {i} \n')

    os.system(f'sbatch ./scripts/{i}.sh')  # execute the script


def main():
    file = open('pmlb.dat', 'rb')
    os.makedirs('./scripts', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    data = pickle.load(file)
    for i, item in enumerate(data):
        print(i)
        assign_jobs(i)
        if i==2:
            return
    #assign_jobs(13)


if __name__ == '__main__':
    main()
