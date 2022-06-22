import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error

from DataGenerators import DataGeneratorSample as DGS
from Losses import CEL
from SparseSetters import SetNoSparse as SNS
from Network import Network
import Bases

import torch
import torch.nn as nn

import time
import sys

logit_test_scores = []
gnb_test_scores = []
tests = [["MSE", mean_squared_error], ["MAE", mean_absolute_error]]
test = ["MSE", mean_squared_error]
layers = [[Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.Exp(),Bases.Log()],
          [Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.Exp(),Bases.Log()],
          [Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.Exp(),Bases.Log()],
          [Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.Exp(),Bases.Log()]]

def occamNetRunFullData(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y, MSELoss):
    learningRate,endTemp,sDev,top,numFuncs,equalization,decay,numEpocs = parameters

    loss = CEL(sDev,int(top*numFuncs),anomWeight = 0)
    sparsifier = SNS()

    n = Network(train_X[0].shape[0],layers,1,sparsifier,loss,learningRate,10,endTemp,equalization)

    batchSize = max(train_X.shape[0]*numFuncs//8000000,1)
    print(batchSize)
    numFuncs2 = numFuncs//batchSize
    print(numFuncs2)
    print(numFuncs-batchSize*numFuncs2)

    trainFunction, valFunction = n.trainFunctionBatch(numEpocs, numFuncs2, batchSize, decay, train_X, train_Y, val_X, val_Y)
    train = MSELoss(train_Y, n.forwardOneFunction(train_X,trainFunction)[:,0]).item()
    val = MSELoss(val_Y, n.forwardOneFunction(val_X,valFunction)[:,0]).item()
    test = MSELoss(test_Y, n.forwardOneFunction(test_X,valFunction)[:,0]).item()
    return (train,val,test,n.applySymbolic(trainFunction)[0],n.applySymbolic(valFunction)[0])

def occamNetGridFullData(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_X = torch.tensor(train_X, device = device).type(torch.float)
    train_Y = torch.tensor(train_Y, device = device).type(torch.float)
    val_X = torch.tensor(val_X, device = device).type(torch.float)
    val_Y = torch.tensor(val_Y, device = device).type(torch.float)
    test_X = torch.tensor(test_X, device = device).type(torch.float)
    test_Y = torch.tensor(test_Y, device = device).type(torch.float)

    MSELoss = nn.MSELoss()

    parameterCombinations = []

    for learningRate in parameters[0]:
        for endTemp in parameters[1]:
            for sDev in parameters[2]:
                for top in parameters[3]:
                    for numFuncs in parameters[4]:
                        for equalization in parameters[5]:
                            for decay in parameters[6]:
                                for epochs in parameters[7]:
                                    parameterCombinations.append((learningRate,endTemp,sDev,top,numFuncs,equalization,decay,epochs))
    
    out = []
    times = []
    i=0
    for param in parameterCombinations:
        print(param)
        startTime = time.perf_counter()
        output = occamNetRunFullData(param,train_X,train_Y,val_X,val_Y,test_X,test_Y,MSELoss)
        endTime = time.perf_counter()
        out.append(output)
        print(endTime-startTime)
        times.append(endTime-startTime)
        print(i)
        i+=1

    train = []
    val = []
    test = []
    trainFunction = []
    valFunction = []
    for data in out:
        trainMSE,valMSE,testMSE,trainFunc,valFunc = data
        train.append(trainMSE)
        val.append(valMSE)
        test.append(testMSE)
        trainFunction.append(trainFunc)
        valFunction.append(valFunc)
    return (parameters,train,val,test,trainFunction,valFunction,times)

def main(i):
    file = open('pmlb.dat', 'rb')
    data = pickle.load(file)
    file.close()
    item = data[i]
    print(item[0])
    trainData = item[1]
    f = open("occamNetPMLB.txt", "a")
    f.write(f"started {i}\n")
    f.close()

    result = occamNetGridFullData([[0.1, 0.5, 1], [10], [0.1, 0.5, 1], [0.1, 0.5, 0.9], [250,1000,4000,16000,64000], [0, 1, 5], [1], [1000]],
                          trainData[0], trainData[1], trainData[2], trainData[3], trainData[4], trainData[5])
    with open(f"occamNetPMLB{i}.dat", "wb") as f:
        pickle.dump(result, f)
    
    print("Done!")


if __name__ == '__main__':
    idx = int(sys.argv[1])
    main(idx)
