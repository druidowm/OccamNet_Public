from DataGenerators import DataGeneratorSample as DGS
from Losses import CEL
from SparseSetters import SetNoSparseNoDuplicates as SNSND
from Network import Network
import Bases
import datetime

import math
import torch
import numpy as np

for i in range(10):
    size = 100
    x = 20*torch.rand(size)-10
    y = 2*x**2+3*x

    X = torch.unsqueeze(x,1)

    loss = CEL(0.01,size//2,anomWeight = 0.2)
    sparsifier = SNSND()

    n = Network(1,[[Bases.Add(),Bases.Add(),Bases.Multiply(),Bases.Multiply()],[Bases.Add(),Bases.Add(),Bases.Multiply(),Bases.Multiply()],[Bases.Add(),Bases.Add(),Bases.Multiply(),Bases.Multiply()]],1,sparsifier,loss,0.01,1,1,0)

    n.trainFunction(3000, size, 1, X, y)
    now2 = datetime.datetime.now()