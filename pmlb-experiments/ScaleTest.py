import pickle
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import argsort

def get_parameter_list(params):
    if len(params)==1:
        return [[item] for item in params[0]]

    parameters = []
    for item in params[0]:
        for item2 in get_parameter_list(params[1:]):
            parameters.append([item]+item2)
    return parameters

def examine_hyperparameters(dataset, paramNumber):
    numFuncs = None
    trainLoss = None
    valLoss = None
    testLoss = None
    successes = 0
    valueList = [i for i in range(81)]
    paramTrains = [[] for i in range(81)]
    for i in range(15):
        try:
            with open(f'{dataset}{i}.dat', 'rb') as f:
                parameters,train,val,test,trainFunc,valFunc,time = pickle.load(f)
                print(parameters)
        
                parameters = get_parameter_list(parameters)

                index = 0
                for i in range(len(parameters)):
                    if parameters[i][4] != 64000:
                        continue
                    print(f"{index} is {parameters[i]}")
                    index+=1

                numFuncs = []
                trainNumFuncs = []
                valNumFuncs = []
                testNumFuncs = []
                timeNumFuncs = []

                index = 0

                for k in range(len(parameters)):
                    if parameters[k][4] != 64000:
                        continue

                    paramTrains[index].append(train[k])
                    #print(paramTrains)
                    index += 1 

                    #print(parameters[k])
                    if parameters[k][paramNumber] in numFuncs:
                        for j in range(len(numFuncs)):
                            if numFuncs[j] == parameters[k][paramNumber]:
                                trainNumFuncs[j].append(train[k])
                                valNumFuncs[j].append(val[k])
                                testNumFuncs[j].append(test[k])
                                timeNumFuncs[j].append(time[k])
                    else:
                        numFuncs.append(parameters[k][paramNumber])
                        trainNumFuncs.append([train[k]])
                        valNumFuncs.append([val[k]])
                        testNumFuncs.append([test[k]])
                        timeNumFuncs.append([time[k]])

                numFuncs = np.array(numFuncs)
                trainNumFuncs = np.array(trainNumFuncs)
                valNumFuncs = np.array(valNumFuncs)
                testNumFuncs = np.array(testNumFuncs)
                timeMeanNumFuncs = np.expand_dims(np.mean(np.array(timeNumFuncs),axis=1),axis=1)
                timeStdNumFuncs = np.expand_dims(np.std(np.array(timeNumFuncs),axis=1),axis=1)

                #print(numFuncs)
                #print(trainNumFuncs)
                #print(valNumFuncs)
                #print(testNumFuncs)
                #print(timeNumFuncs)

                bestTrain = np.expand_dims(np.min(trainNumFuncs, axis=1),axis=1)
                bestValPos = np.expand_dims(np.argmin(valNumFuncs, axis=1),axis=1)
                bestVal = np.take_along_axis(valNumFuncs,bestValPos,axis=1)
                bestTest = np.take_along_axis(testNumFuncs,bestValPos,axis=1)

                pos=1
                midTrain = np.take_along_axis(trainNumFuncs, np.argsort(trainNumFuncs,axis=1)[:,pos:pos+1], axis=1)
                midVal = np.take_along_axis(valNumFuncs, np.argsort(valNumFuncs,axis=1)[:,pos:pos+1], axis=1)

                #print(bestTrain)
                #print(bestVal)
                #print(bestTest)

                print(f"successes = {successes}")
                if successes==0:
                    numFuncs = numFuncs

                    trainLoss = bestTrain
                    valLoss = bestVal
                    testLoss = bestTest

                    midTrainLoss = midTrain
                    midValLoss = midVal

                    timeMean = timeMeanNumFuncs
                    timeStd = timeStdNumFuncs
                else:
                    trainLoss = np.concatenate((trainLoss,bestTrain),axis=1)
                    valLoss = np.concatenate((valLoss,bestVal),axis=1)
                    testLoss = np.concatenate((testLoss,bestTest),axis=1)

                    midTrainLoss = np.concatenate((midTrainLoss,midTrain),axis=1)
                    midValLoss = np.concatenate((midValLoss,midVal),axis=1)

                    timeMean = np.concatenate((timeMean,timeMeanNumFuncs),axis=1)
                    timeStd = np.concatenate((timeStd,timeStdNumFuncs),axis=1)

                successes += 1
        except IOError as e:
            pass

    print(trainLoss[-1]-trainLoss[0])
    plt.plot(numFuncs,np.mean(trainLoss,axis=1))
    plt.show()

    for i in range(len(paramTrains[0])):
        print([item[i] for item in paramTrains])
        plt.plot(valueList,np.array([item[i] for item in paramTrains])/np.min([item[i] for item in paramTrains]))
    
    plt.scatter(valueList,np.mean(paramTrains,axis=1)/np.min(np.mean(paramTrains,axis=1)))
    plt.plot([0,80],[1,1])
    plt.show()

def main(dataset):
    numFuncs = None
    trainLoss = None
    valLoss = None
    testLoss = None
    successes = 0
    for i in range(15):
        try:
            with open(f'{dataset}{i}.dat', 'rb') as f:
                parameters,train,val,test,trainFunc,valFunc,time = pickle.load(f)
        
                parameters = get_parameter_list(parameters)

                numFuncs = []
                trainNumFuncs = []
                valNumFuncs = []
                testNumFuncs = []
                timeNumFuncs = []


                for k in range(len(parameters)):
                    #print(parameters[k])
                    if parameters[k][4] in numFuncs:
                        for j in range(len(numFuncs)):
                            if numFuncs[j] == parameters[k][4]:
                                trainNumFuncs[j].append(train[k])
                                valNumFuncs[j].append(val[k])
                                testNumFuncs[j].append(test[k])
                                timeNumFuncs[j].append(time[k])
                    else:
                        numFuncs.append(parameters[k][4])
                        trainNumFuncs.append([train[k]])
                        valNumFuncs.append([val[k]])
                        testNumFuncs.append([test[k]])
                        timeNumFuncs.append([time[k]])

                numFuncs = np.array(numFuncs)
                trainNumFuncs = np.array(trainNumFuncs)
                valNumFuncs = np.array(valNumFuncs)
                testNumFuncs = np.array(testNumFuncs)
                timeMeanNumFuncs = np.expand_dims(np.mean(np.array(timeNumFuncs),axis=1),axis=1)
                timeStdNumFuncs = np.expand_dims(np.std(np.array(timeNumFuncs),axis=1),axis=1)

                #print(numFuncs)
                #print(trainNumFuncs)
                #print(valNumFuncs)
                #print(testNumFuncs)
                #print(timeNumFuncs)

                bestTrain = np.expand_dims(np.min(trainNumFuncs, axis=1),axis=1)
                bestValPos = np.expand_dims(np.argmin(valNumFuncs, axis=1),axis=1)
                bestVal = np.take_along_axis(valNumFuncs,bestValPos,axis=1)
                bestTest = np.take_along_axis(testNumFuncs,bestValPos,axis=1)

                pos=1
                midTrain = np.take_along_axis(trainNumFuncs, np.argsort(trainNumFuncs,axis=1)[:,pos:pos+1], axis=1)
                midVal = np.take_along_axis(valNumFuncs, np.argsort(valNumFuncs,axis=1)[:,pos:pos+1], axis=1)

                #print(bestTrain)
                #print(bestVal)
                #print(bestTest)

                print(f"successes = {successes}")
                if successes==0:
                    numFuncs = numFuncs

                    trainLoss = bestTrain
                    valLoss = bestVal
                    testLoss = bestTest

                    midTrainLoss = midTrain
                    midValLoss = midVal

                    timeMean = timeMeanNumFuncs
                    timeStd = timeStdNumFuncs
                else:
                    trainLoss = np.concatenate((trainLoss,bestTrain),axis=1)
                    valLoss = np.concatenate((valLoss,bestVal),axis=1)
                    testLoss = np.concatenate((testLoss,bestTest),axis=1)

                    midTrainLoss = np.concatenate((midTrainLoss,midTrain),axis=1)
                    midValLoss = np.concatenate((midValLoss,midVal),axis=1)

                    timeMean = np.concatenate((timeMean,timeMeanNumFuncs),axis=1)
                    timeStd = np.concatenate((timeStd,timeStdNumFuncs),axis=1)

                successes += 1
        except IOError as e:
            pass

    trainLoss1 = trainLoss[0].copy()
    valLoss1 = valLoss[0].copy()

    trainLoss /= trainLoss1
    valLoss /= valLoss1
    testLoss /= testLoss[0]
    
    midTrainLoss /= trainLoss1
    midValLoss /= valLoss1

    trainLossMean = np.mean(trainLoss,axis=1)
    valLossMean = np.mean(valLoss,axis=1)
    testLossMean = np.mean(testLoss,axis=1)

    trainLossSdev = np.std(trainLoss,axis=1)
    valLossSdev = np.std(valLoss,axis=1)
    testLossSdev = np.std(testLoss,axis=1)

    fig,axes = plt.subplots(2,2)

    for i in range(trainLoss.shape[1]):
        axes[0][0].errorbar(numFuncs,trainLoss[:,i],yerr=(midTrainLoss[:,i]-trainLoss[:,i]),lolims=True)
    axes[0][0].set_xscale("log",base=2)
    axes[0][0].set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    axes[0][0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    axes[1][0].errorbar(numFuncs,valLossMean,yerr=valLossSdev)
    axes[1][0].set_xscale("log",base=2)
    axes[1][0].set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    axes[1][0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    axes[0][1].errorbar(numFuncs,testLossMean,yerr=testLossSdev)
    axes[0][1].set_xscale("log",base=2)
    axes[0][1].set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    axes[0][1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    for i in range(timeMean.shape[1]):
        axes[1][1].errorbar(numFuncs,timeMean[:,i],yerr=timeStd[:,i])
    axes[1][1].set_xscale("log",base=2)
    axes[1][1].set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    axes[1][1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1][1].set_yscale("log",base=2)

    plt.show()

examine_hyperparameters("occamNetPMLB",3)
main("occamNetPMLB")