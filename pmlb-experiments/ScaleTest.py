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

def examine_hyperparameters(dataset):
    numFuncs = None
    trainLoss = None
    successes = 0
    valueList = [i for i in range(81)]
    paramTrains = [[] for i in range(81)]
    for i in range(15):
        try:
            with open(f'{dataset}{i}.dat', 'rb') as f:
                parameters,train,_,_,_,_,_ = pickle.load(f)
                print(parameters)
                print(train)
        
                parameters = get_parameter_list(parameters)
                print(parameters)
                
                parameters2 = []
                index = 0
                for i in range(len(parameters)):
                    if parameters[i][4] == 64000:
                        print(f"{index}: {parameters[i]}")
                        index += 1
                        parameters2.append(parameters[i])

                numFuncs = []
                trainNumFuncs = []

                index = 0
                for k in range(len(parameters)):
                    print(parameters[k][4])
                    if parameters[k][4] == 64000:

                        print(index)
                        print(k)
                        paramTrains[index].append(train[k])
                        index+=1

                        if parameters[k][4] in numFuncs:
                            for j in range(len(numFuncs)):
                                if numFuncs[j] == parameters[k][4]:
                                    trainNumFuncs[j].append(train[k])
                        else:
                            numFuncs.append(parameters[k][4])
                            trainNumFuncs.append([train[k]])

                numFuncs = np.array(numFuncs)
                trainNumFuncs = np.array(trainNumFuncs)

                bestTrain = np.expand_dims(np.min(trainNumFuncs, axis=1),axis=1)

                print(f"successes = {successes}")
                if successes==0:
                    numFuncs = numFuncs

                    trainLoss = bestTrain
                else:
                    trainLoss = np.concatenate((trainLoss,bestTrain),axis=1)

                successes += 1
        except IOError as e:
            pass

    #print(trainLoss[-1]-trainLoss[0])
    #plt.plot(numFuncs,np.mean(trainLoss,axis=1))
    #plt.show()

    avg = 0

    count = 0
    for i in range(len(paramTrains[0])):
        #print([item[i] for item in paramTrains])
        plot_values = np.array([item[i] for item in paramTrains])/np.min([item[i] for item in paramTrains])
        plt.plot(valueList,plot_values)
        avg += plot_values
        plt.plot([0,80],[1,1])
        plt.show()
        print(f"Dataset {i+1}")
        if np.abs(plot_values[20]-1)<0.1 or np.abs(plot_values[47]-1)<0.1 or np.abs(plot_values[74]-1)<0.1:
            count+=1
            print(f"Count: {count}")

        if not (np.abs(plot_values[20]-1)<0.001 or np.abs(plot_values[47]-1)<0.001 or np.abs(plot_values[74]-1)<0.001):
            print(np.argmin(plot_values))
            print(parameters2[np.argmin(plot_values)])
        #print(min(plot_values[20],plot_values[47],plot_values[74]))
    print(f"Count: {count}")
    plt.plot(valueList,avg/15)
    
    #plt.scatter(valueList,np.mean(paramTrains,axis=1)/np.min(np.mean(paramTrains,axis=1)))
    plt.plot([0,80],[1,1])
    plt.show()


def sortByNumFuncs(parameters,numFuncsLoc,train,val,test,time):
    numFuncs = []
    trainNumFuncs = []
    valNumFuncs = []
    testNumFuncs = []
    timeNumFuncs = []

    for k in range(len(parameters)):
        #print(parameters[k])
        if parameters[k][numFuncsLoc] in numFuncs:
            for j in range(len(numFuncs)):
                if numFuncs[j] == parameters[k][numFuncsLoc]:
                    trainNumFuncs[j].append(train[k])
                    valNumFuncs[j].append(val[k])
                    testNumFuncs[j].append(test[k])
                    timeNumFuncs[j].append(time[k])
        else:
            numFuncs.append(parameters[k][numFuncsLoc])
            trainNumFuncs.append([train[k]])
            valNumFuncs.append([val[k]])
            testNumFuncs.append([test[k]])
            timeNumFuncs.append([time[k]])

    numFuncs = np.array(numFuncs)
    trainNumFuncs = np.array(trainNumFuncs)
    valNumFuncs = np.array(valNumFuncs)
    testNumFuncs = np.array(testNumFuncs)
    timeNumFuncs = np.array(timeNumFuncs)

    return numFuncs,trainNumFuncs,valNumFuncs,testNumFuncs,timeNumFuncs


def concatenateData(prevData, data):
    if type(prevData) == type(None):
        return data
    else:
        return np.concatenate((prevData,data),axis=1)

def loadResults(dataset):
    numFuncs = None

    bestTrainLoss = None
    bestValLoss = None
    bestValTestLoss = None

    meanTrainLoss = None
    meanValLoss = None
    meanTestLoss = None

    midTrainLoss = None
    midValLoss = None

    timeMean = None
    timeStd = None

    for i in range(15):
        try:
            with open(f'{dataset}{i}.dat', 'rb') as f:
                parameters,train,val,test,trainFunc,valFunc,time = pickle.load(f)
                #print(train)
            
                parameters = get_parameter_list(parameters)

                if dataset[0] == "o":
                    numFuncsLoc = 4
                else:
                    numFuncsLoc = 0

                numFuncs,trainNumFuncs,valNumFuncs,testNumFuncs,timeNumFuncs = sortByNumFuncs(parameters,numFuncsLoc,train,val,test,time)
                
                """
                if i==3:
                    if dataset[0] == "o":
                        index = 0
                        for j in range(len(parameters)):
                            if parameters[j][4] == 64000:
                                if index in [20,47,74]:
                                    print(trainFunc[j])
                                    print(train[j])
                                index += 1
                                print(index)
                    
                    else:
                        for j in range(len(parameters)):
                            if parameters[j][0] == 4000:
                                print(trainFunc[j])
                                print(train[j])
                """
                

                if dataset[0] == "o":
                    trainNumFuncs = trainNumFuncs[:,(20,47,74)]
                    valNumFuncs = valNumFuncs[:,(20,47,74)]
                    testNumFuncs = testNumFuncs[:,(20,47,74)]

                timeMeanNumFuncs = np.expand_dims(np.mean(timeNumFuncs,axis=1),axis=1)
                timeStdNumFuncs = np.expand_dims(np.std(timeNumFuncs,axis=1),axis=1)

                bestTrain = np.expand_dims(np.min(trainNumFuncs, axis=1),axis=1)
                bestValPos = np.expand_dims(np.argmin(valNumFuncs, axis=1),axis=1)
                #print(valNumFuncs)
                #print(bestValPos)
                bestVal = np.take_along_axis(valNumFuncs,bestValPos,axis=1)
                bestTest = np.take_along_axis(testNumFuncs,bestValPos,axis=1)

                pos=1
                midTrain = np.take_along_axis(trainNumFuncs, np.argsort(trainNumFuncs,axis=1)[:,pos:pos+1], axis=1)
                midVal = np.take_along_axis(valNumFuncs, np.argsort(valNumFuncs,axis=1)[:,pos:pos+1], axis=1)

                meanTrain = np.expand_dims(np.mean(trainNumFuncs, axis=1),axis=1)
                meanVal = np.expand_dims(np.mean(valNumFuncs, axis=1),axis=1)
                meanTest = np.expand_dims(np.mean(testNumFuncs, axis=1),axis=1)

                numFuncs = numFuncs

                bestTrainLoss = concatenateData(bestTrainLoss, bestTrain)
                bestValLoss = concatenateData(bestValLoss, bestVal)
                bestValTestLoss = concatenateData(bestValTestLoss, bestTest)

                meanTrainLoss = concatenateData(meanTrainLoss, meanTrain)
                meanValLoss = concatenateData(meanValLoss, meanVal)
                meanTestLoss = concatenateData(meanTestLoss, meanTest)

                midTrainLoss = concatenateData(midTrainLoss, midTrain)
                midValLoss = concatenateData(midValLoss, midVal)
                timeMean = concatenateData(timeMean, timeMeanNumFuncs)
                timeStd = concatenateData(timeStd, timeStdNumFuncs)

        except IOError as e:
            print("could not find file")
            pass

    return numFuncs,(bestTrainLoss,bestValLoss,bestValTestLoss),(meanTrainLoss,meanValLoss,meanTestLoss),(midTrainLoss,midValLoss),(timeMean,timeStd)

def plotTimeScaling(models,grid,counter):
    fig = plt.figure(figsize=(6,4.2))
    ax = plt.subplot(grid[0],grid[1],counter)

    for model in models:
        numFuncs,_,_,_,times = loadResults(model)
        timeMean,timeStd = times

        occamnet = (model[0]=="o")

        #if occamnet:
        #    timeMean *= 81
        #    timeStd *= 81
        #else:
        #    timeMean *= 3
        #    timeStd *= 3
    
        label = "OccamNet" if occamnet else "Eplex"

        linestyle = "-" if occamnet else ":"

        color = "tab:blue" if occamnet else "tab:orange"

        ax.errorbar([],[], yerr = [], label=label, color = color)
        for i in range(timeMean.shape[1]):
            ax.errorbar(numFuncs,timeMean[:,i],yerr = timeStd[:,0], color = color, alpha=0.5)

    ax.set_xscale("log",base=2)
    ax.set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yscale("log",base=2)
    #ax.set_title("Mean Run Time")
    ax.set_xlabel("Functions Sampled per Epoch", fontsize = 14)
    ax.set_ylabel("Mean Run Time (s)", fontsize = 14)
    ax.legend(fontsize = "x-large")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig("TimeScalingResults.pdf")

def main(models, grid, i, counter):
    #fig,axes = plt.subplots(2,2)
    ax = plt.subplot(grid[0],grid[1],counter)
    
    counter+=1

    for model in models:
        numFuncs,bestLosses,meanLosses,midLosses,times = loadResults(model)
        bestTrainLoss,bestValLoss,bestValTestLoss = bestLosses
        meanTrainLoss,meanValLoss,meanTestLoss = meanLosses
        midTrainLoss,midValLoss = midLosses
        timeMean,timeStd = times

        #print("HERE")
        #print(bestTrainLoss)
        
        bestTrainLoss1 = bestTrainLoss[0].copy()
        bestValLoss1 = bestValLoss[0].copy()

        #no_outliers = (np.sum(testLoss,axis=0) <= 1)
        #testLoss = testLoss[:,no_outliers]
        
        midTrainLoss /= bestTrainLoss1
        midValLoss /= bestValLoss1

        occamnet = (model[0]=="o")

        label = "OccamNet" if occamnet else "Eplex"
        linestyle = "-" if occamnet else "-."
        color = "tab:blue" if occamnet else "tab:orange"

        if occamnet:
            timeMean *= 3#81
        else:
            timeMean *= 3

        ax.plot(timeMean[:,i], bestTrainLoss[:,i], linestyle = linestyle, color = "tab:blue", label = f"{label} Train")
        ax.plot(timeMean[:,i], bestValLoss[:,i], linestyle = linestyle, color = "limegreen", label = f"{label} Validation")
        ax.plot(timeMean[:,i], bestValTestLoss[:,i], linestyle = linestyle, color = "tab:orange", label = f"{label} Test")

    ax.set_xscale("log")
    ax.set_title(f"Dataset {i+1}")

    ax.set_ylabel("MSE Loss")

    #ax.set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    ax.set_xlabel("Run Time (s)")

    ax.set_yscale("log")

    if i==0:
        ax.legend()

    #plt.show()

    return counter


    """
        label = "OccamNet" if model[0]=="o" else "Eplex"
        axes[0][0].plot(numFuncs, bestTrainLoss[:,i])
        axes[1][0].plot(numFuncs, bestValTestLoss[:,i])
        axes[0][1].plot(numFuncs, bestValLoss[:,i])
        axes[1][1].plot(numFuncs, timeMean[:,i],label=label)

    axes[0][0].set_xscale("log",base=2)
    axes[0][0].set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    axes[0][0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0][0].set_title("Mean Training Loss")
    axes[0][0].set_ylabel("MSE Loss")

    axes[1][0].set_xscale("log",base=2)
    axes[1][0].set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    axes[1][0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1][0].set_title("Mean Testing Loss")
    axes[1][0].set_xlabel("Functions Sampled per Epoch")
    axes[1][0].set_ylabel("MSE Loss")

    axes[0][1].set_xscale("log",base=2)
    axes[0][1].set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    axes[0][1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0][1].set_title("Mean Validation Loss")
    axes[0][1].set_ylabel("MSE Loss")

    axes[1][1].set_xscale("log",base=2)
    axes[1][1].set_xticks([250,500,1000,2000,4000,8000,16000,32000,64000])
    axes[1][1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1][1].set_yscale("log",base=2)
    axes[1][1].set_title("Mean Run Time")
    axes[1][1].set_xlabel("Functions Sampled per Epoch")
    axes[1][1].set_ylabel("Time (s)")
    axes[1][1].legend()

    plt.show()"""

#examine_hyperparameters("occamNet_scale_results/occamNetPMLB")

#"""
plotTimeScaling(["genetic_scale_results/geneticPMLB","occamNet_scale_results/occamNetPMLB"],(1,1),1)

"""
fig = plt.figure(figsize=(18,20))
grid = (5,3)
counter = 1
for i in range(15):
    counter = main(["occamNet_scale_results/occamNetPMLB","genetic_scale_results/geneticPMLB"],grid,i,counter)

plt.tight_layout()
plt.savefig("LossScalingResults2.pdf")


fig = plt.figure(figsize=(12,8))
grid = (2,2)
counter = main(["occamNet_scale_results/occamNetPMLB","genetic_scale_results/geneticPMLB"],grid,3,1)
counter = main(["occamNet_scale_results/occamNetPMLB","genetic_scale_results/geneticPMLB"],grid,11,counter)
counter = main(["occamNet_scale_results/occamNetPMLB","genetic_scale_results/geneticPMLB"],grid,13,counter)
#plotTimeScaling(["genetic_scale_results/geneticPMLB","occamNet_scale_results/occamNetPMLB"],grid,counter)

ax = plt.subplot(2,2,4)
ax.plot([],[],linestyle = "-", color = "tab:blue", label = "OccamNet Training MSE")
ax.plot([],[],linestyle = "-", color = "limegreen", label = "OccamNet Validation MSE")
ax.plot([],[],linestyle = "-", color = "tab:orange", label = "OccamNet Testing MSE")
ax.plot([],[],linestyle = "-.", color = "tab:blue", label = "Eplex Training MSE")
ax.plot([],[],linestyle = "-.", color = "limegreen", label = "Eplex Validation MSE")
ax.plot([],[],linestyle = "-.", color = "tab:orange", label = "Eplex Testing MSE")
ax.axis("off")
ax.legend(loc = "center",fontsize = "xx-large")

plt.tight_layout(w_pad = 2, h_pad = 2)
plt.savefig("LossScalingResults_Small2.pdf")#"""