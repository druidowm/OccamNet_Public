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

def occamNetGrid(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_X = torch.tensor(train_X, device = device).type(torch.float)
    train_Y = torch.tensor(train_Y, device = device).type(torch.float)
    val_X = torch.tensor(val_X, device = device).type(torch.float)
    val_Y = torch.tensor(val_Y, device = device).type(torch.float)
    test_X = torch.tensor(test_X, device = device).type(torch.float)
    test_Y = torch.tensor(test_Y, device = device).type(torch.float)

    MSELoss = nn.MSELoss()


    bestTrainNet = None
    bestTrainFunction = None
    bestTrain = float("inf")
    bestTrainParams = []

    bestValNet = None
    bestValFunction = None
    bestVal = float("inf")
    bestValParams = []

    startTime = time.perf_counter()
    i=0
    for learningRate in parameters[0]:
        for endTemp in parameters[1]:
            for sDev in parameters[2]:
                for top in parameters[3]:
                    for percentBatchSize in parameters[4]:
                        for equalization in parameters[5]:
                            for decay in parameters[6]:

                                batchSize = int(train_X.shape[0]*percentBatchSize)
                                batchesPerEpoch = int(train_X.shape[0]/batchSize)
                                numFuncs = int(15000000/batchSize)

                                loss = CEL(sDev,int(top*numFuncs),anomWeight = 0)
                                sparsifier = SNS()

                                n = Network(train_X[0].shape[0],layers,1,sparsifier,loss,learningRate,10,endTemp,equalization)

                                trainFunction,valFunction = n.trainFunction(int(1000/batchesPerEpoch), numFuncs, decay, train_X, train_Y, val_X, val_Y)
                                
                                train = MSELoss(train_Y, n.forwardOneFunction(train_X,trainFunction)[:,0]).item()
                                print("train: "+str(train))
                                if (train<bestTrain):
                                    bestTrainNet = n
                                    bestTrainFunction = trainFunction
                                    bestTrain = train
                                    bestTrainParams = [learningRate, endTemp, sDev, top, numFuncs, equalization]

                                val = MSELoss(val_Y, n.forwardOneFunction(val_X,valFunction)[:,0]).item()
                                print("validation: "+str(val))
                                if (val<bestVal):
                                    bestValNet = n
                                    bestValFunction = valFunction
                                    bestVal = val
                                    bestValParams = [learningRate, endTemp, sDev, top, numFuncs, equalization]
                                i+=1
                                print(i)

    endTime = time.perf_counter()
    avgTime = (endTime-startTime)/i

    bestTest = MSELoss(test_Y,bestValNet.forwardOneFunction(test_X,bestValFunction)[:,0]).item()

    return (bestTrain,bestTrainNet.applySymbolic(bestTrainFunction),bestTrainParams,
            bestVal,bestValNet.applySymbolic(bestValFunction),bestValParams,
            bestTest,avgTime)

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

    result = occamNetGrid([[0.5, 1], [10], [0.1, 0.5, 1], [0.1, 0.5, 0.9], [1], [0, 1, 5], [1]],
                          trainData[0], trainData[1], trainData[2], trainData[3], trainData[4], trainData[5])
    f = open("occamNetPMLB.txt", "a")
    f.write("Dataset:\t\tTraining Score\t\tTraining Result\t\tTraining Params\t\tValidation Score\t\tValidation Result\t\tValidation Params\t\tTest Result\t\tAverage Time\n")
    f.write(item[0]+",\t\t"+str(result[0])+",\t\t"+str(result[1])+",\t\t"+str(result[2])+",\t\t"+str(result[3])+",\t\t"+str(result[4])+",\t\t"+str(result[5])+",\t\t"+str(result[6])+",\t\t"+str(result[7])+"\n")
    f.close()


if __name__ == '__main__':
    idx = int(sys.argv[1])
    main(idx)
