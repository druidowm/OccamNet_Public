import multiprocessing
import torch
import torch.nn as nn
import numpy as np
import Bases
from Losses import CrossEntropyLoss
from Network import NetworkConstants, ActivationLayer
from SparseSetters import SetPartialSparse as SPS
from SparseSetters import SetNoSparse as SNS
import argparse
import datetime
import pickle

from torch.distributions import Categorical
import torch.nn.functional as F

import argparse

def func(x):
    return 10.5*torch.pow(x,3.1)

if __name__ == '__main__':
    fileName = "constantPower"

    times = []
    dataSize = 200

    for i in range(10):
        now1 = datetime.datetime.now()
        loss = CrossEntropyLoss(0.0005,10)
        sparsifier = SNS()

        x = torch.arange(0.1,1,0.001).unsqueeze(1)
        y = func(x)

        n = NetworkConstants(1,[[Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.AddConstant(),Bases.MultiplyConstant(),Bases.PowerConstant()],[Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.AddConstant(),Bases.MultiplyConstant(),Bases.PowerConstant()]],1,sparsifier,loss,0.01,0.05,1,10, 1)

        n.setConstants([0 for j in range(n.totalConstants)])

        n.trainFunction(2000, max(1,dataSize/200), 200, 1.0, x, y, useMultiprocessing = True, numProcesses = 14)

        now2 = datetime.datetime.now()
        times.append(now2-now1)
        print(times[-1])
        
    

    outFile = open(fileName+"Times.list","wb")
    pickle.dump(times,outFile)
