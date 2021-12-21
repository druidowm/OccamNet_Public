from pmlb import fetch_data, regression_dataset_names
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
from numpy import random
#import array

#import xgboost as xgb

#from deap import algorithms
#from deap import base
#from deap import creator
#from deap import tools

#import matplotlib.pyplot as plt

#import seaborn as sb

from GeneticAlgorithm import geneticRun, geneticRunFullData
from Feynman import aiFeynmanRun as AFR
from XGBoostClass import XGBobj
from DataGenerators import DataGeneratorSample as DGS
from Losses import CEL
from SparseSetters import SetNoSparse as SNS
from Network import Network
import Bases

import torch
import torch.nn as nn

import operator
import math
import random
#from deap import gp

from pmlb import fetch_data, regression_dataset_names
import time

#import matplotlib.pyplot as plt

#import ray

import argparse

#import multiprocessing
#from functools import partial



logit_test_scores = []
gnb_test_scores = []
tests = [["MSE",mean_squared_error],["MAE",mean_absolute_error]]
test = ["MSE",mean_squared_error]
layers = [[Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.Exp(),Bases.Log()],
          [Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.Exp(),Bases.Log()],
          [Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.Exp(),Bases.Log()],
          [Bases.Add(),Bases.Subtract(),Bases.Multiply(),Bases.Divide(),Bases.Sin(),Bases.Cos(),Bases.Exp(),Bases.Log()]]


def xgbGrid(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    bestValXGB = None
    bestVal = float("inf")
    bestValParams = []

    bestTrainXGB = None
    bestTrain = float("inf")
    bestTrainParams = []

    startTime = time.perf_counter()

    i = 0
    for max_depth in parameters[0]:
        for subsample in parameters[1]:
            for gamma in parameters[2]:
                for learning_rate in parameters[3]:
                    for n_estimators in parameters[4]:
                        print("run "+str(i))
                        xgb = XGBobj(max_depth = max_depth,
                                        gamma = gamma,
                                        learning_rate = learning_rate,
                                        n_estimators = n_estimators,
                                        subsample = subsample)

                        xgb.train(train_X, train_Y, val_X, val_Y, 100000)

                        train = xgb.valTest(train_X, train_Y)
                        val = xgb.valTest(val_X, val_Y)

                        print("training: "+str(train))
                        print("validation: "+str(val))
                        if (val<bestVal):
                            bestValXGB = xgb
                            bestVal = val
                            bestValParams = [max_depth, subsample, gamma, learning_rate,
                                            n_estimators]

                        if (train<bestTrain):
                            bestTrainXGB = xgb
                            bestTrain = train
                            bestTrainParams = [max_depth, subsample, gamma, learning_rate,
                                            n_estimators]
                        i+=1

    endTime = time.perf_counter()
    avgTime = (endTime-startTime)/i

    return (bestTrain,bestTrainParams,
            bestVal,bestValParams,
            bestValXGB.test(test_X, test_Y, test),avgTime)

def occamNetRun(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y, MSELoss):
    learningRate,endTemp,sDev,top,numFuncs,equalization,decay,numEpocs = parameters

    loss = CEL(sDev,int(top*numFuncs),anomWeight = 0)
    sparsifier = SNS()

    n = Network(train_X[0].shape[0],layers,1,sparsifier,loss,learningRate,10,endTemp, equalization)

    trainFunction, valFunction = n.trainFunction(numEpocs, numFuncs, decay, train_X, train_Y, val_X, val_Y)
    train = MSELoss(train_Y, n.forwardOneFunction(train_X,trainFunction)[:,0]).item()
    val = MSELoss(val_Y, n.forwardOneFunction(val_X,valFunction)[:,0]).item()
    test = MSELoss(test_Y, n.forwardOneFunction(test_X,valFunction)[:,0]).item()
    return (train,val,test,trainFunction,valFunction)

def occamNetRunFullData(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y, MSELoss):
    learningRate,endTemp,sDev,top,numFuncs,equalization,decay,numEpocs = parameters

    loss = CEL(sDev,int(top*numFuncs),anomWeight = 0)
    sparsifier = SNS()

    n = Network(train_X[0].shape[0],layers,1,sparsifier,loss,learningRate,10,endTemp, equalization)

    trainFunction, valFunction = n.trainFunction(numEpocs, numFuncs, decay, train_X, train_Y, val_X, val_Y)
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
    for param in parameters:
        startTime = time.perf_counter()
        output = occamNetRun(param,train_X,train_Y,val_X,val_Y,test_X,test_Y,MSELoss)
        endTime = time.perf_counter()
        out.append(output)
        times.append(endTime-startTime)

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

def geneticGridFullData(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    train_X = [train_X[:,i] for i in range(train_X.shape[1])]
    val_X = [val_X[:,i] for i in range(val_X.shape[1])]
    test_X = [test_X[:,i] for i in range(test_X.shape[1])]

    params = []
    for popSize in parameters[0]:
        for epochs in parameters[1]:
            for constraint in parameters[2]:
                for crossover in parameters[3]:
                    params.append([popSize,epochs,constraint,crossover])
    outs = [geneticRunFullData(param, train_X, train_Y, val_X, val_Y) for param in params]

    train = []
    val = []
    test = []
    trainFunction = []
    valFunction = []
    times = []
    for data in outs:
        e,runTime = data
        times.append(runTime)
        train.append(e.bestTrain)
        trainFunction.append(e.bestTrainFunction)
        val.append(e.bestVal)
        valFunction.append(e.bestValFunction)
        test.append(e.bestTest)

    return (parameters,train,val,test,trainFunction,valFunction,times)

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


def occamNetGridCPU(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y):
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
                    for functions in parameters[4]:
                        for equalization in parameters[5]:
                            for decay in parameters[6]:
                                numFuncs = functions

                                loss = CEL(sDev,int(top*numFuncs),anomWeight = 0)
                                sparsifier = SNS()

                                n = Network(train_X[0].shape[0],layers,1,sparsifier,loss,learningRate,10,endTemp, equalization)

                                trainFunction, valFunction = n.trainFunction(int(1000000/numFuncs), numFuncs, decay, train_X, train_Y, val_X, val_Y)
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

def geneticGrid(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    bestTrainTree = None
    bestTrainParams = None
    bestTrain = float("inf")
    
    bestValTree = None
    bestValParams = None
    bestVal = float("inf")

    train_X = [train_X[:,i] for i in range(train_X.shape[1])]
    val_X = [val_X[:,i] for i in range(val_X.shape[1])]
    test_X = [test_X[:,i] for i in range(test_X.shape[1])]

    params = []
    for popSize in parameters[0]:
        for constraint in parameters[1]:
            for crossover in parameters[2]:
                params.append([popSize,constraint,crossover])

    startTime = time.perf_counter()
    outs = [geneticRun(param, train_X, train_Y, val_X, val_Y) for param in params]

    endTime = time.perf_counter()
    avgTime = (endTime-startTime)/len(outs)

    for i in range(len(outs)):
        if (outs[i].bestVal<bestVal):
            bestValTree = outs[i]
            bestVal = outs[i].bestVal
            bestValParams = params[i]

        if (outs[i].bestTrain<bestTrain):
            bestTrainTree = outs[i]
            bestTrain = outs[i].bestTrain
            bestTrainParams = params[i]

    bestTest = bestValTree.getError(bestValTree.bestValFunction, test_X, test_Y)

    return (bestTrain, bestTrainTree.bestTrainFunction, bestTrainParams,
            bestVal, bestValTree.bestValFunction, bestValParams,
            bestTest, avgTime)

def feynmanGrid(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    bestTrainFunction = None
    bestTrainParams = None
    bestTrain = float("inf")
    
    bestValFunction = None
    bestValParams = None
    bestVal = float("inf")

    train_X = [train_X[:,i] for i in range(train_X.shape[1])]
    val_X = [val_X[:,i] for i in range(val_X.shape[1])]
    test_X = [test_X[:,i] for i in range(test_X.shape[1])]

    params = []
    for symbolicTime in parameters[0]:
        for polyfit_deg in parameters[1]:
            for NN_epochs in parameters[2]:
                params.append([symbolicTime,polyfit_deg,NN_epochs])

    startTime = time.perf_counter()
    outs = [AFR(train_X, train_Y, val_X, val_Y, param) for param in params]

    endTime = time.perf_counter()
    avgTime = (endTime-startTime)/len(outs)

    for i in range(len(outs)):
        if (outs[i][3]<bestVal):
            bestValFunction = outs[i][2]
            bestVal = outs[i][3]
            bestValParams = params[i]

        if (outs[i][1]<bestTrain):
            bestTrainFunction = outs[i][0]
            bestTrain = outs[i][1]
            bestTrainParams = params[i]

    bestTest = mean_squared_error(test_Y, bestValFunction[0](*test_X))

    return (bestTrain, bestTrainFunction[1], bestTrainParams,
            bestVal, bestValFunction[1], bestValParams,
            bestTest, avgTime)


def main(task, process):
    i=0
    j=0
    for dataset in regression_dataset_names:
        X, y = fetch_data(dataset, return_X_y=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.expand_dims(y, axis = 1)
        scaler = StandardScaler()
        y = scaler.fit_transform(y).squeeze()

        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = .4, random_state = 42)
        val_X, test_X, val_Y, test_Y = train_test_split(test_X, test_Y, test_size = .5, random_state = 24)

        if (train_X.shape[0]<=1000):
            print(dataset)
            print(X.shape)
            if i<15:
                print("here")
                if task == "OccamNet":
                    if process != None:
                        if int(i/2) == process:
                            result = occamNetGridCPU([[0.5,1],[10],[0.5,1],[0.1,0.5,0.9],[500,1000,2000],[1, 5],[1]],train_X, train_Y, val_X, val_Y, test_X, test_Y)
                            f = open("occamNetPMLB"+str(process)+".txt","a")
                            f.write("Dataset:\t\tTraining Score\t\tTraining Result\t\tTraining Params\t\tValidation Score\t\tValidation Result\t\tValidation Params\t\tTest Result\t\tAverage Time\n")
                            f.write(dataset+",\t\t"+str(result[0])+",\t\t"+str(result[1])+",\t\t"+str(result[2])+",\t\t"+str(result[3])+",\t\t"+str(result[4])+",\t\t"+str(result[5])+",\t\t"+str(result[6])+",\t\t"+str(result[7])+"\n")
                            f.close()
                    else:
                        #result = occamNetGrid([[0.5,1],[10],[0.1],[0.1],[1],[5],[1]],train_X, train_Y, val_X, val_Y, test_X, test_Y)
                        #result = occamNetGrid([[0.5,1,2],[10],[0.1,0.5,1],[0.1,0.5,0.9],[1],[0, 1, 4, 5],[0.99,1]],train_X, train_Y, val_X, val_Y, test_X, test_Y)
                        #result = occamNetGridCPU([[0.5,1],[10],[0.5,1],[0.1,0.5,0.9],[500,1000,2000],[1, 5],[1]],train_X, train_Y, val_X, val_Y, test_X, test_Y)
                        f = open("occamNetPMLB.txt","a")
                        f.write("Dataset:\t\tTraining Score\t\tTraining Result\t\tTraining Params\t\tValidation Score\t\tValidation Result\t\tValidation Params\t\tTest Result\t\tAverage Time\n")
                        f.write(dataset+",\t\t"+str(result[0])+",\t\t"+str(result[1])+",\t\t"+str(result[2])+",\t\t"+str(result[3])+",\t\t"+str(result[4])+",\t\t"+str(result[5])+",\t\t"+str(result[6])+",\t\t"+str(result[7])+"\n")
                        f.close()

                if task == "XGBoost":
                    if process != None:
                        if int(i/2) == process:
                            result = xgbGrid([[6],[0.5,0.75,1],[0,0.1,0.2,0.3,0.4],[0.0001,0.01,0.05,0.1,0.2],[10,50,100,250,500,1000]],train_X, train_Y, val_X, val_Y, test_X, test_Y)
                            f = open("xgbPMLB"+str(process)+".txt","a")
                            f.write("Dataset:\t\tTraining Score\t\tTraining Params\t\tValidation Score\t\tValidation Params\t\tTest Result\t\tAverage Time\n")
                            f.write(dataset+",\t\t"+str(result[0])+",\t\t"+str(result[1])+",\t\t"+str(result[2])+",\t\t"+str(result[3])+",\t\t"+str(result[4])+",\t\t"+str(result[5])+"\n")
                            f.close()
                    else:
                        result = xgbGrid([[6],[0.5,0.75,1],[0,0.1,0.2,0.3,0.4],[0.0001,0.01,0.05,0.1,0.2],[10,50,100,250,500,1000]],train_X, train_Y, val_X, val_Y, test_X, test_Y)
                        f = open("xgbPMLB.txt","a")
                        f.write("Dataset:\t\tTraining Score\t\tTraining Params\t\tValidation Score\t\tValidation Params\t\tTest Result\t\tAverage Time\n")
                        f.write(dataset+",\t\t"+str(result[0])+",\t\t"+str(result[1])+",\t\t"+str(result[2])+",\t\t"+str(result[3])+",\t\t"+str(result[4])+",\t\t"+str(result[5])+"\n")
                        f.close()

                if task == "Eplex":
                    if process != None:
                        if int(i/2) == process:
                            result = geneticGrid([[500,1000,2000],[4],[0.2,0.5,0.8]],train_X, train_Y, val_X, val_Y, test_X, test_Y)
                            f = open("eplexPMLB"+str(process)+".txt","a")
                            f.write("Dataset:\t\tTraining Score\t\tTraining Result\t\tTraining Params\t\tValidation Score\t\tValidation Result\t\tValidation Params\t\tTest Result\t\tAverage Time\n")
                            f.write(dataset+",\t\t"+str(result[0])+",\t\t"+str(result[1])+",\t\t"+str(result[2])+",\t\t"+str(result[3])+",\t\t"+str(result[4])+",\t\t"+str(result[5])+",\t\t"+str(result[6])+",\t\t"+str(result[7])+"\n")
                            f.close()

                    else:
                        result = geneticGrid([[500,1000,2000],[4],[0.2,0.5,0.8]],train_X, train_Y, val_X, val_Y, test_X, test_Y)
                        f = open("eplexPMLB.txt","a")
                        f.write("Dataset:\t\tTraining Score\t\tTraining Result\t\tTraining Params\t\tValidation Score\t\tValidation Result\t\tValidation Params\t\tTest Result\t\tAverage Time\n")
                        f.write(dataset+",\t\t"+str(result[0])+",\t\t"+str(result[1])+",\t\t"+str(result[2])+",\t\t"+str(result[3])+",\t\t"+str(result[4])+",\t\t"+str(result[5])+",\t\t"+str(result[6])+",\t\t"+str(result[7])+"\n")
                        f.close()
                
                if task == "Feynman":
                    if train_X.shape[1] <= 10:
                        result = feynmanGrid([[100],[0,1],[100,500]],train_X,train_Y,val_X,val_Y,test_X,test_Y)
                        f = open("feynmanPMLB.txt","a")
                        f.write("Dataset:\t\tTraining Score\t\tTraining Result\t\tTraining Params\t\tValidation Score\t\tValidation Result\t\tValidation Params\t\tTest Result\t\tAverage Time\n")
                        f.write(dataset+",\t\t"+str(result[0])+",\t\t"+str(result[1])+",\t\t"+str(result[2])+",\t\t"+str(result[3])+",\t\t"+str(result[4])+",\t\t"+str(result[5])+",\t\t"+str(result[6])+",\t\t"+str(result[7])+"\n")
                        f.close()
                    else:
                        j-=1
            i+=1
            j+=1


parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task', type=str)
parser.add_argument('--process', dest='process', type=int)

args = parser.parse_args()

#ray.init()

if __name__ == '__main__':
    main(args.task,args.process)