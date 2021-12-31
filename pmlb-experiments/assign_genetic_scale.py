import pickle
import os
import time
from multiprocessing import Pool


def assign_jobs(i):
    os.system(f'python3 run_genetic_scale.py --dataset {i}')  # execute the script


def main():
    file = open('pmlb.dat', 'rb')

    """n=1#len(pickle.load(file))

    startTime = time.perf_counter()

    with Pool(n) as p:
        p.map(assign_jobs, [i for i in range(1)])
    
    endTime = time.perf_counter()
    timeDiff = endTime-startTime
    print(timeDiff)"""

    n=len(pickle.load(file))

    with Pool(n) as p:
        p.map(assign_jobs, [i for i in range(8)])

    with Pool(n) as p:
        p.map(assign_jobs, [i for i in range(8,n)])


if __name__ == '__main__':
    main()