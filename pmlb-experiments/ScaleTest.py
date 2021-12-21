import pickle
import numpy as np
from matplotlib import pyplot as plt

def main(dataset):
    numFuncs = None
    trainLoss = None
    valLoss = None
    testLoss = None
    for i in range(15):
        with open(f'{dataset}Scale{i}.dat', 'rb') as f:
            parameters,train,val,test,trainFunc,valFunc,time = pickle.loads(f)
    
            numFuncs = []
            trainNumFuncs = []
            valNumFuncs = []
            testNumFuncs = []
            timeNumFuncs = []

            for k in range(len(parameters)):
                if parameters[k][4] in numFuncs:
                    found = False
                    for j in range(len(numFuncs)):
                        if numFuncs[j] == parameters[k][4]:
                            trainNumFuncs[j].append(train[k])
                            valNumFuncs[j].append(val[k])
                            testNumFuncs[j].append(test[k])
                            timeNumFuncs[j].append(time[k])
                    if not found:
                        numFuncs.append(parameters[k][4])
                        trainNumFuncs.append([train[k]])
                        valNumFuncs.append([val[k]])
                        testNumFuncs.append([test[k]])
                        timeNumFuncs.append([time[k]])

            numFuncs = np.array(numFuncs)
            trainNumFuncs = np.array(trainNumFuncs)
            valNumFuncs = np.array(valNumFuncs)
            testNumFuncs = np.array(testNumFuncs)
            timeNumFuncs = np.sum(np.array(timeNumFuncs),axis=1)

            bestTrain = np.min(trainNumFuncs, axis=1)
            bestValPos = np.argmin(valNumFuncs, axis=1)
            bestVal = valNumFuncs[bestValPos]
            bestTest = testNumFuncs[bestValPos]

            if i==0:
                numFuncs = numFuncs
                trainLoss = np.expand_dims(bestTrain,axis=1)
                valLoss = np.expand_dims(bestVal,axis=1)
                testLoss = np.expand_dims(bestTest,axis=1)
            else :
                trainLoss = np.concatenate(trainLoss,np.expand_dims(bestTrain,axis=1),axis=1)
                valLoss = np.concatenate(valLoss,np.expand_dims(bestVal,axis=1),axis=1)
                testLoss = np.concatenate(testLoss,np.expand_dims(bestTest,axis=1),axis=1)

    trainLoss /= trainLoss[0]
    valLoss /= valLoss[0]
    testLoss /= testLoss[0]
    trainLoss = np.sum(trainLoss,axis=1)
    valLoss = np.sum(valLoss,axis=1)
    testLoss = np.sum(testLoss,axis=1)

    plt.plot(numFuncs,trainLoss)
    plt.plot(numFuncs,valLoss)
    plt.plot(numFuncs,testLoss)


                