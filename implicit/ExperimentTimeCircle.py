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
    y = torch.sqrt(1-x[0]**2)
    rand = 2*torch.round(torch.rand([y.shape[0],1]))-1
    return y*rand

if __name__ == '__main__':
    fileName = "circle"

    times = []
    dataSize = 200

    for i in range(10):
        now1 = datetime.datetime.now()
        dg4 = ImplicitFunctionDataGeneratorSample(2,min(200,dataSize),(-1,1),dataSize,implicitFunction,1.0)

        loss = CELActivationRegularization(0.01,10,0.7,0.3,0.15,0.1,verbosity=0)
        sparsifier = SNS()
        n = NetworkConstants(2,[[Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos()],[Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos()]],1,sparsifier,loss,0.01,0.01,1,10)

        n.setConstants([0 for j in range(n.totalConstants)])

        n.trainFunction(dg4, 2000, max(1,dataSize/200), 200, "gradient", useMultiprocessing = True, saveState = True, saveStateSteps = 5, trackHighestProb = True, plot = True, plotName = "circlePlot")

        now2 = datetime.datetime.now()
        times.append(now2-now1)
        print(times[-1])
        
    

    outFile = open(fileName+"Times.list","wb")
    pickle.dump(times,outFile)