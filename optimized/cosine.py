from DataGenerators import DataGeneratorSample as DGS
from Losses import CEL, Adaptive, Adaptive2,Adaptive4
from SparseSetters import SetNoSparseNoDuplicates as SNSND
from Network import Network
import Bases
import datetime

import math
import torch
import numpy as np
import matplotlib.pyplot as plt

times = []

if __name__ == "__main__":
    for i in range(10):
        size= 200
        x = 200*torch.rand(size)-100
        y = torch.cos(x)

        x = torch.unsqueeze(x,1)
        x2 = torch.full((size,1),math.pi)
        x3 = torch.full((size,1),2)
        X = torch.cat((x,x2,x3),1)

        now1 = datetime.datetime.now()
        loss = CEL(0.01,50,anomWeight = 0.2)
        loss = Adaptive4(0.01,375,anomWeight=0)
        sparsifier = SNSND()

        print("HERE!!!!!!!!!!!!!")

        n = Network(3,[[Bases.Add(),Bases.Sin(),Bases.Divide()],[Bases.Add(),Bases.Sin(),Bases.Divide()],[Bases.Add(),Bases.Sin(),Bases.Divide()]],1,sparsifier,loss,0.01,1,10,5)

        n.trainFunction(3000, 400, 1, X, y)
        now2 = datetime.datetime.now()
        times.append(now2-now1)

        path,prob = n.getPathMaxProb()

        x = torch.arange(-10,10,0.1)
        x = torch.unsqueeze(x,1)
        x2 = torch.full((x.shape[0],1),math.pi)
        x3 = torch.full((x.shape[0],1),2)
        X = torch.cat((x,x2,x3),1)

        out = n.forwardOneFunction(X,path)[:,0]
        indices = torch.argsort(x[:,0])
        plt.plot(x[:,0][indices],torch.cos(x[:,0][indices]))
        plt.plot(x[:,0][indices],out[indices])
        plt.show()

print(times)
print(np.mean(times))