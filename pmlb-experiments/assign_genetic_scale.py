import pickle
import os
from multiprocessing import Pool


def assign_jobs(i):
    os.system(f'python3 run_genetic_scale.py --dataset {i}')  # execute the script


def main():
    file = open('pmlb.dat', 'rb')

    n=len(pickle.load(file))

    with Pool(n) as p:
        p.map(assign_jobs, [i for i in range(n)])

    """for i, item in enumerate(data):
        print(i)
        assign_jobs(i)
        if i==1:
            return"""


if __name__ == '__main__':
    main()