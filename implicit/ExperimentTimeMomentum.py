import torch
import torch.nn as nn
import numpy as np
import Bases
from DataGenerators import FunctionDataGenerator,ImplicitFunctionDataGenerator,MultivariateFunctionDataGenerator,ImplicitFunctionDataGeneratorSample
from Losses import CrossEntropyLoss,CELTrivialRegularization,CELTrivialConstantRegularization,CELFlagRegularization,CELActivationRegularization
from NetworkRegularization import NetworkConstants, ActivationLayer
from SparseSetters import SetPartialSparse as SPS
from SparseSetters import SetNoSparse as SNS
import argparse
import datetime
import pickle

from torch.distributions import Categorical
import torch.nn.functional as F

import argparse

def implicitFunction(x):
    return x[0]*x[1]/x[2]

if __name__ == '__main__':
    fileName = "momentum"

    times = []
    dataSize = 200

    for i in range(10):
        now1 = datetime.datetime.now()
        dg4 = ImplicitFunctionDataGeneratorSample(4,min(200,dataSize),(-10,10),dataSize,implicitFunction,0.0)

        loss = CELActivationRegularization(0.01,10,0.7,0.3,0.15,0.1,verbosity=0)
        sparsifier = SNS()
        
        n = NetworkConstants(4,[[Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos()],[Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos()]],1,sparsifier,loss,0.01,0.01,1,10)

        n.setConstants([0 for j in range(n.totalConstants)])

        n.trainFunction(dg4, 2000, max(1,dataSize/200), 200, "gradient", useMultiprocessing = True, saveState = True, saveStateSteps = 5, trackHighestProb = True, plot = True, plotName = "momentumPlot")

        now2 = datetime.datetime.now()
        times.append(now2-now1)
        print(times[-1])
        
    

    outFile = open(fileName+"Times.list","wb")
    pickle.dump(times,outFile)