import torch
import torch.nn as nn
import numpy as np
import Bases
from DataGenerators import FunctionDataGenerator,ImplicitFunctionDataGenerator,MultivariateFunctionDataGenerator,ImplicitFunctionDataGeneratorSample,FunctionDataGeneratorSample
from Losses import CrossEntropyLoss,CELTrivialRegularization,CELTrivialConstantRegularization,CELFlagRegularization,CELActivationRegularization
from NetworkRegularization import NetworkConstants, ActivationLayer
from SparseSetters import SetPartialSparse as SPS
from SparseSetters import SetNoSparse as SNS
from SparseSetters import SetNoSparseNoDuplicates as SNSND
import math
import argparse
import datetime
import pickle

from torch.distributions import Categorical
import torch.nn.functional as F

import argparse

def implicitFunction(x):
    return torch.exp(x)

if __name__ == '__main__':
    fileName = "taylor"

    times = []
    dataSize = 200

    for i in range(10):
        now1 = datetime.datetime.now()
        dg4 = FunctionDataGeneratorSample(min(200,dataSize),dataSize,(0,1),implicitFunction,[1])

        loss = CELActivationRegularization(0.05,10,0.7,0.3,0.05,0.03,verbosity=0)
        sparsifier = SNS()
        
        n = NetworkConstants(2,[[Bases.Add(),Bases.MultiplyConstant(),Bases.PowerConstant()],[Bases.Add(),Bases.MultiplyConstant(),Bases.PowerConstant()],[Bases.Add(),Bases.MultiplyConstant(),Bases.PowerConstant()]],1,sparsifier,loss,0.01,0.05,1,10)

        n.setConstants([0 for j in range(n.totalConstants)])

        n.trainFunction(dg4, 10000, max(1,dataSize/200), 200, "gradient", useMultiprocessing = True, saveState = True, saveStateSteps = 5, trackHighestProb = True, plot = True, plotName = "taylorPlot")

        now2 = datetime.datetime.now()
        times.append(now2-now1)
        print(times[-1])
        
    

    outFile = open(fileName+"Times.list","wb")
    pickle.dump(times,outFile)
