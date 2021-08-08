from DataGenerators import DataGeneratorSample as DGS
from Losses import CEL
from SparseSetters import SetNoSparseNoDuplicates as SNSND
from Network import Network
import Bases
import datetime

import math
import torch
import numpy as np

times = []

for i in range(10):
    x = 200*torch.rand(400)-100
    y = torch.cos(x)

    x = torch.unsqueeze(x,1)
    x2 = torch.full((400,1),math.pi)
    x3 = torch.full((400,1),2)
    X = torch.cat((x,x2,x3),1)

    now1 = datetime.datetime.now()
    loss = CEL(0.01,50,anomWeight = 0.2)
    sparsifier = SNSND()

    n = Network(3,[[Bases.Add(),Bases.Sin(),Bases.Divide()],[Bases.Add(),Bases.Sin(),Bases.Divide()],[Bases.Add(),Bases.Sin(),Bases.Divide()]],1,sparsifier,loss,0.01,1,10,1)

    n.trainFunction(3000, 400, 1, X, y)
    now2 = datetime.datetime.now()
    times.append(now2-now1)

print(times)
print(np.mean(times))