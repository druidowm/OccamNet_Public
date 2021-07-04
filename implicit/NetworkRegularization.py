from numpy.lib.npyio import save
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import Bases
from DataGenerators import FunctionDataGenerator,ImplicitFunctionDataGenerator
from Losses import CrossEntropyLoss,CELFlagRegularization
import argparse
import sympy as sp
from sympy import *
import time
from matplotlib import rc, rcParams
from matplotlib import patches as patch

from SympyTest import sympyEquals

from torch.distributions import Categorical
import torch.nn.functional as F

import multiprocessing
from multiprocessing import Value
from functools import partial

import datetime
import time

import pickle

class ActivationLayer:
    def __init__(self, activations):
        self.activations = activations
        self.totalInputs = 0
        self.totalConstants = 0
        self.constantActivations = []
        for item in activations:
            self.totalInputs += item.numInputs
        for item in activations:
            if isinstance(item,Bases.BaseWithConstants):
                self.totalConstants += 1
                self.constantActivations.append(item)

    def apply(self, input, flags):
        output = torch.empty((input.shape[0],len(self.activations)))
        newFlags = []
        i = 0 
        for j in range(0,len(self.activations)):
            numInputs = self.activations[j].numInputs
            output[:,j],flag = self.activations[j].getOutput(input[:,i:i+numInputs])

            for k in range(i,i+numInputs):
                flag += flags[k]
            newFlags.append(flag)

            i+=numInputs
        return (output,newFlags)

    def applySymbolic(self, input):
        output = []
        index = 0
        for item in self.activations:
            output.append(item.getSymbolicOutput(input[index:index+item.numInputs]))
            index += item.numInputs
        return output

    def applySymbolicConstant(self, input, trivial):
        output = []
        index = 0
        totalTrivial = []
        for item in self.activations:
            out,numTrivial = item.getSymbolicOutputConstant(input[index:index+item.numInputs])
            output.append(out)

            for i in range(index,index + item.numInputs):
                numTrivial += trivial[i]

            totalTrivial.append(numTrivial)

            index += item.numInputs
        return (output,totalTrivial)
    
    def setConstants(self, constants):
        for i in range(len(self.constantActivations)):
            self.constantActivations[i].getConstant(constants[i])

    def setSymbolicConstants(self, startNum):
        for i in range(len(self.constantActivations)):
            self.constantActivations[i].getSymbolicConstant("c_"+str(startNum+i))
        return startNum + len(self.constantActivations)

    def getNumConstants(self, constants):
        index = 0
        numConst = []
        for item in self.activations:
            num = 0
            if item in self.constantActivations:
                num += 1

            for i in range(index,index + item.numInputs):
                num += constants[i]

            numConst.append(num)

            index += item.numInputs

        return numConst

    def getNumActivations(self, activations):
        index = 0
        numAct = []
        for item in self.activations:
            num = 1

            for i in range(index,index + item.numInputs):
                num += activations[i]

            numAct.append(num)

            index += item.numInputs

        return numAct

class NetworkConstants(nn.Module):
    def __init__(self, inputSize, activationLists, outputSize, sparseSetter, loss, learningRate, constantLearningRate, temp, endTemp, recursiveDepth = 1, skipConnections = True):
        super().__init__()
        self.skipConnections = skipConnections
        self.inputSize = inputSize
        self.activationLayers,self.sparse = sparseSetter.getActivationsSparsity(inputSize,activationLists,outputSize)
        self.outputSize = outputSize
        self.temp = temp
        self.endTemp = endTemp
        self.recursiveDepth = recursiveDepth
        self.totalConstants = 0
        self.loss = loss
        self.learningRate = learningRate
        self.constantLearningRate = constantLearningRate
        self.layers = [nn.Linear(inputSize,self.activationLayers[0].totalInputs,bias=False)]
        if skipConnections:
            prevLayerSize = self.inputSize
            for i in range(0,len(self.activationLayers)-1):
                prevLayerSize+=len(self.activationLayers[i].activations)
                self.layers.append(nn.Linear(prevLayerSize,self.activationLayers[i+1].totalInputs, bias = False))
            self.layers.append(nn.Linear(len(self.activationLayers[-1].activations)+prevLayerSize,outputSize, bias = False))
        else:
            for i in range(0,len(self.activationLayers)-1):
                self.layers.append(nn.Linear(len(self.activationLayers[i].activations),self.activationLayers[i+1].totalInputs, bias = False))
            self.layers.append(nn.Linear(len(self.activationLayers[-1].activations),outputSize, bias = False))

        self.layers = nn.ModuleList(self.layers)

        self.plot()

        with torch.no_grad():
            for layer in self.layers:
                layer.weight[:,:] = 0
            self.equalizeWeights()

        for layer in self.activationLayers:
            self.totalConstants += layer.totalConstants

        self.constants = nn.parameter.Parameter(torch.rand([self.totalConstants], dtype=torch.float))

        constNum = 0
        for layer in self.activationLayers:
            constNum = layer.setSymbolicConstants(constNum)

        manager = multiprocessing.Manager()
        self.testedFunctions = manager.dict()
        self.symbolicTestedFunctions = manager.dict()
        self.setConstants(self.constants)
        self.timesTested = manager.Value("i",0)

    def equalizeWeights(self):
        path = []
        self.layers[0].weight[self.sparse[0]] = -20*self.temp
        weight = F.softmax(self.layers[0].weight/self.temp, dim=1)
        prob2 = torch.tensor([torch.max(weight[i]) for i in range(self.activationLayers[0].totalInputs)], dtype=torch.float)


        for i in range(0,len(self.layers)-1):
            prob3 = torch.ones([len(self.activationLayers[i].activations)],dtype = torch.float)
            index = 0
            for j in range(0,prob3.shape[0]):
                for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                    prob3[j] *= prob2[k]
                index+= self.activationLayers[i].activations[j].numInputs
            
            if i == 0:
                if self.skipConnections:
                    prob = torch.cat([prob3,torch.ones([self.inputSize], dtype = torch.float)], dim = 0)
                else:
                    prob = prob3
            else:
                if self.skipConnections:
                    prob = torch.cat([prob3,prob], dim = 0)
                else:
                    prob = prob3



            for l in range(self.layers[i+1].weight.shape[0]):
                numProbs = []
                probWeights = []
                for j in range(prob.shape[0]):
                    if self.sparse[i+1][l,j]:
                        found = False
                        for k in range(len(probWeights)):
                            if abs(probWeights[k]/prob[j]-1) < 0.01:
                                numProbs[k] += 1
                                found = True

                        if not found:
                            numProbs.append(1)
                            probWeights.append(prob[j])


                probMatrix = [numProbs]
                for j in range(1,len(probWeights)):
                    matLayer = [0 for i in range(len(probWeights))]
                    matLayer[0] = probWeights[0]
                    matLayer[j] = -probWeights[j]
                    probMatrix.append(matLayer)

                probMatrix = torch.tensor(probMatrix, dtype = torch.float)

                outVec = [0 for i in range(len(probWeights))]
                outVec[0] = 1
                outVec = torch.tensor(outVec,dtype = torch.float)

                weightVals = torch.matmul(probMatrix.inverse(),outVec)


                if i == len(self.layers)-2:
                    bot = math.exp(1/self.endTemp)/weightVals[0]
                    weightVals = self.endTemp*torch.log(bot*weightVals)
                else:
                    bot = math.exp(1/self.temp)/weightVals[0]
                    weightVals = self.temp*torch.log(bot*weightVals)


                for k in range(prob.shape[0]):
                    if self.sparse[i+1][l,k]:
                        for j in range(0,len(probWeights)):
                            if abs(prob[k]/probWeights[j]-1)<0.01:
                                self.layers[i+1].weight[l,k] = weightVals[j]
                    else:
                        if i == len(self.layers)-2:
                            self.layers[i+1].weight[l,k] = -20*self.endTemp+torch.min(weightVals)
                        else:
                            self.layers[i+1].weight[l,k] = -20*self.temp+torch.min(weightVals)

            if i == len(self.layers)-2:
                weight = F.softmax(self.layers[i+1].weight/self.endTemp, dim=1)
            else:
                weight = F.softmax(self.layers[i+1].weight/self.temp, dim=1)


            prob2 = torch.empty([self.layers[i+1].weight.shape[0]],dtype = torch.float)
            for j in range(weight.shape[0]):
                done = False
                for k in range(weight.shape[1]):
                    if not done:
                        if self.sparse[i+1][j,k]:
                            prob2[j] = prob[k]*weight[j,k]
                            done = True


    def plotNode(self,pos,text = None):
        if text != None:
            plt.text(pos[0],pos[1],text,fontsize = 10,horizontalalignment = "center",verticalalignment = "center",zorder=3)
        theta = np.arange(0,2*math.pi+2*math.pi/300,2*math.pi/300)
        plt.fill(pos[0]+15*np.cos(theta),pos[1]+22*np.sin(theta),"r",zorder=2)

    def plotLayer(self,prevXpos,xpos,numNodes,connections = None):
        for i in range(numNodes):
            ypos = 100*i
            if connections != None:
                for k in range(connections.shape[1]):
                    if connections[i,k]:
                        plt.plot([prevXpos, xpos],[100*k,ypos],"g",zorder = 1)


            self.plotNode((xpos,ypos))

    def plotActivationLayer(self,prevXpos,xpos,activations,numSkipNodes, connections = None):
        index = 0
        for i in range(len(activations)):
            ypos = 100*i
            xdelta = -100
            for j in range(activations[i].numInputs):
                ydelta = 50*j-25*activations[i].numInputs
                plt.plot([xpos+xdelta,xpos],[ypos+ydelta,ypos],"b",zorder = 1)
                if connections != None:
                    for k in range(connections.shape[1]):
                        if connections[index,k]:
                            plt.plot([prevXpos, xpos+xdelta],[100*k,ypos+ydelta],"g",zorder = 1)

                self.plotNode((xpos+xdelta,ypos+ydelta))
                index += 1


            self.plotNode((xpos,ypos),activations[i].getLatex())

        for i in range(numSkipNodes):
            ypos = 100*(i+len(activations))
            plt.plot([prevXpos,xpos],[100*i,ypos],"k--",zorder = 1)
            self.plotNode((xpos,ypos))

    
    def plot(self):
        self.plotLayer(0,0,self.inputSize)
        numSkip = self.inputSize
        for i in range(len(self.activationLayers)):
            xpos = i*500+500
            self.plotActivationLayer(xpos-500,xpos,self.activationLayers[i].activations,numSkip,self.sparse[i])
            numSkip += len(self.activationLayers[i].activations)
        
        self.plotLayer(500*len(self.activationLayers),500*len(self.activationLayers)+500,self.outputSize,self.sparse[len(self.activationLayers)])
        plt.show()



    def applySymbolic(self, path):
        input = ["x_"+str(i) for i in range(self.inputSize)]
        for i in range(len(path)-1):
            inter = []
            for j in range(path[i].shape[0]):
                inter.append(input[path[i][j]])
            input = self.activationLayers[i].applySymbolic(inter)+input
        inter = []
        for j in range(path[-1].shape[0]):
            inter.append("y_"+str(j)+"="+input[path[-1][j]])
        return inter
        
    def applySymbolicConstant(self, path):

        input = ["x_"+str(i) for i in range(self.inputSize)]
        totalTrivial = [0 for i in range(self.inputSize)]

        for i in range(len(path)-1):

            inter = []
            interTrivial = []

            for j in range(path[i].shape[0]):
                inter.append(input[path[i][j]])
                interTrivial.append(totalTrivial[path[i][j]])
            
            inter,interTrivial = self.activationLayers[i].applySymbolicConstant(inter,interTrivial)
            input = inter+input
            totalTrivial = interTrivial+totalTrivial

        inter = []
        numTrivial = 0
        for j in range(path[-1].shape[0]):
            inter.append("y_"+str(j)+"="+input[path[-1][j]])
            numTrivial += totalTrivial[path[-1][j]]

        return (inter,numTrivial)

    def getNumConstants(self, path):
        numConstants = [0 for i in range(self.inputSize)]

        for i in range(len(path)-1):

            inter = []

            for j in range(path[i].shape[0]):
                inter.append(numConstants[path[i][j]])
            
            inter = self.activationLayers[i].getNumConstants(inter)
            numConstants = inter+numConstants

        num = 0
        for j in range(path[-1].shape[0]):
            num += numConstants[path[-1][j]]

        return num

    def getNumActivations(self, path):
        numAct = [0 for i in range(self.inputSize)]
        for i in range(len(path)-1):
            inter = []
            for j in range(path[i].shape[0]):
                inter.append(numAct[path[i][j]])
            numAct = self.activationLayers[i].getNumActivations(inter)+numAct
        totalAct = 0
        for j in range(path[-1].shape[0]):
            totalAct += numAct[path[-1][j]]
        return totalAct

    def setConstants(self, constants):
        i = 0
        for layer in self.activationLayers:
            if layer.totalConstants > 0:
                numConst = layer.totalConstants
                layer.setConstants(constants[i:i+numConst])
                i += numConst
    
    def getTrainingSamples(self, sampleSize):
        paths = []
        probs = torch.ones((sampleSize,self.inputSize), dtype = torch.float)
        for i in range(0,len(self.layers)-1):
            weight = F.softmax(self.layers[i].weight/self.temp, dim=1)

            try:
                path = Categorical(weight).sample([sampleSize])
            except:
                None

            paths.append(path)

            probs2 = torch.gather(probs, 1, path) * torch.gather(weight.T, 0, path)

            prob = torch.ones((sampleSize,len(self.activationLayers[i].activations)))
            index = 0
            for j in range(0,prob.shape[1]):
                for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                    prob[:,j] *= probs2[:,k]
                index+= self.activationLayers[i].activations[j].numInputs


            if self.skipConnections:
                probs = torch.cat([prob,probs], dim = 1)
            else:
                probs = prob



        weight = F.softmax(self.layers[-1].weight/self.endTemp, dim=1)

        path = Categorical(weight).sample([sampleSize])
        paths.append(path)


        probs = torch.gather(probs, 1, path) * torch.gather(weight.T, 0, path)
        prob = torch.prod(probs,1)
        return (paths,prob)
    
    def getPathArgmax(self):
        path = []
        prob = torch.ones([self.inputSize], dtype = torch.float)

        for i in range(0,len(self.layers)-1):
            weight = F.softmax(self.layers[i].weight/self.temp, dim=1)
            path.append(torch.argmax(weight, dim = 1))

            probs = torch.gather(prob, 0, path[i]) * torch.gather(weight.T, 0, path[i].unsqueeze(0))[0]

            prob2 = torch.ones([len(self.activationLayers[i].activations)])
            index = 0
            for j in range(0,prob2.shape[0]):
                for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                    prob2[j] *= probs[k]
                index+= self.activationLayers[i].activations[j].numInputs

            if self.skipConnections:
                prob = torch.cat([prob2,prob], dim = 0)
            else:
                prob = prob2


        weight = F.softmax(self.layers[-1].weight/self.endTemp, dim=1)

        path.append(torch.argmax(weight,dim=1))


        prob = torch.gather(prob, 0, path[-1]) * torch.gather(weight.T, 0, path[-1].unsqueeze(0))[0]
        prob = torch.prod(prob)
        return (path,prob)

    def getPathMaxProb(self):
        path = []
        prob = torch.ones([self.inputSize], dtype = torch.float)

        for i in range(0,len(self.layers)-1):
            weight = F.softmax(self.layers[i].weight/self.temp, dim=1)
            splProbs = prob.unsqueeze(0).repeat((weight.shape[0],1))
            splProbs = weight*splProbs
            path.append(torch.argmax(splProbs, dim = 1))


            probs = torch.gather(splProbs.T, 0, path[i].unsqueeze(0))[0]

            prob2 = torch.ones([len(self.activationLayers[i].activations)])
            index = 0
            for j in range(0,prob2.shape[0]):
                for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                    prob2[j] *= probs[k]
                index+= self.activationLayers[i].activations[j].numInputs

            if self.skipConnections:
                prob = torch.cat([prob2,prob], dim = 0)
            else:
                prob = prob2

        weight = F.softmax(self.layers[-1].weight/self.endTemp, dim=1)
        splProbs = prob.unsqueeze(0).repeat((weight.shape[0],1))
        splProbs = weight*splProbs
        path.append(torch.argmax(splProbs, dim = 1))

        probs = torch.gather(splProbs.T, 0, path[-1].unsqueeze(0))[0]
        prob = torch.prod(probs)
        return (path,prob)


    def getProb(self, path):
        with torch.no_grad():
            probs = torch.ones((self.inputSize,), dtype = torch.float)
            for i in range(0,len(self.layers)-1):
                weight = F.softmax(self.layers[i].weight/self.temp, dim=1)

                probs2 = probs[path[i]] * torch.gather(weight.T, 0, path[i].unsqueeze(0))[0]

                prob = torch.ones((len(self.activationLayers[i].activations),))
                index = 0
                for j in range(0,prob.shape[0]):
                    for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                        prob[j] *= probs2[k]
                    index+= self.activationLayers[i].activations[j].numInputs


                probs = torch.cat([prob,probs], dim = 0)



            weight = F.softmax(self.layers[-1].weight/self.endTemp, dim=1)

            probs = probs[path[-1]] * torch.gather(weight.T, 0, path[-1].unsqueeze(1))
            prob = torch.prod(probs,1)

            return prob


    def getTrivialOperations(self, symbolic):
        if symbolic in self.symbolicTestedFunctions:
            return self.symbolicTestedFunctions[symbolic]

        simple = sympify(symbolic[symbolic.find("=")+1:])
        original = sympify(symbolic[symbolic.find("=")+1:],evaluate = False)
        if sympyEquals(simple, original,True):
            if sympyEquals(simple, original,False):
                self.symbolicTestedFunctions[symbolic]=0
                return 0
            else:
                self.symbolicTestedFunctions[symbolic]=0.1
                return 0.1

        self.symbolicTestedFunctions[symbolic]=1
        return 1

    def forward(self, input, path):
        outputs = torch.empty((input.shape[0],self.outputSize,self.recursiveDepth), dtype = torch.float)
        path = [item.unsqueeze(0).repeat(input.shape[0], 1) for item in path]

        for i in range(self.recursiveDepth):
            flags = [0 for i in range(self.inputSize)]
            for j in range(len(self.layers)-1):
                img = torch.gather(input, 1, path[j])
                interFlags = []

                for k in range(path[j].shape[1]):
                    interFlags.append(flags[path[j][0,k]])

                inter,interFlags = self.activationLayers[j].apply(img,interFlags)
                input = torch.cat([inter,input], dim = 1)

                flags = interFlags + flags

            input = torch.gather(input, 1, path[-1])


            flagSum = 0
            for j in range(path[-1].shape[1]):
                flagSum += flags[path[-1][j]]

            outputs[:,:,i] = input
        return (outputs,flagSum)

    def fitConstantsGradient(self, input, path, y):
        MSELoss = nn.MSELoss()
        constantList = torch.empty((self.recursiveDepth,self.totalConstants), dtype = torch.float)
        outputs = torch.empty((input.shape[0],y.shape[1],self.recursiveDepth), dtype = torch.float)
        for j in range(self.recursiveDepth):
            self.constants = nn.Parameter(torch.rand([self.totalConstants], dtype=torch.float))
            optimizer = torch.optim.Adam(self.parameters(), lr=self.constantLearningRate)
            self.setConstants(self.constants)

            output,flags = self.forward(input,path)
            output = output[:,:,j]

            losses = []
            values = [[] for i in range(self.constants.shape[0])]

            count = 0
            while torch.any(output!=output) and count<100:
                self.constants = nn.Parameter(torch.rand([self.totalConstants], dtype=torch.float))
                self.setConstants(self.constants)
                output,flags = self.forward(input,path)
                output = output[:,:,j]
                count+=1
            
            if count >= 100:
                outputs[:,:,j] = output
                constantList[j,:] = self.constants
                break

            for i in range(2000):
                lossVal = MSELoss(y,output)
                losses.append(lossVal)

                for k in range(len(values)):
                    values[k].append(self.constants[k].item())

                optimizer.zero_grad()
                lossVal.backward()
        
                if torch.all(torch.abs(self.constants.grad)<0.0001) or torch.any(self.constants.grad!=self.constants.grad):
                    break

                optimizer.step()
                output,flags = self.forward(input,path)
                output = output[:,:,j]

                if torch.any(output!=output):
                    break

            outputs[:,:,j] = output
            constantList[j,:] = self.constants
        return (constantList.detach(),outputs.detach(),flags)

    def fitConstantsEvolutionary(self, input, path, y):
        pass

    def forwardFitConstants(self, path, input = None, y = None, method = "gradient"):
        if self.constants.shape[0]==0:
            equations,numTrivial = self.applySymbolicConstant(path)
            if equations[0] == "y_0=(x_0*x_1)" or equations[0] == "y_0=(x_1*x_0)":
                self.timesTested.value += 1

            equationsNum = self.applySymbolic(path)

            for item in equationsNum:
                numTrivial += self.getTrivialOperations(item)

            outputs,flags = self.forward(input,path)
            return (outputs,flags,numTrivial)

        else:
            equations,numTrivial = self.applySymbolicConstant(path)
            if equations[0] in self.testedFunctions:
                self.setConstants(self.testedFunctions[equations[0]])
                equations,numTrivial = self.applySymbolicConstant(path)
                if equations[0] == "y_0=(x_0*x_1)" or equations[0] == "y_0=(x_1*x_0)":
                    self.timesTested.value += 1

                equationsNum = self.applySymbolic(path)

                for item in equationsNum:
                    numTrivial += self.getTrivialOperations(item)

                outputs,flags = self.forward(input,path)
                return (outputs,flags,numTrivial)
            else:
                if method == "gradient":
                    constantList,outputs,flags = self.fitConstantsGradient(input,[item.detach() for item in path],y)
                    equations,numTrivial = self.applySymbolicConstant(path)

                    if equations[0] == "y_0=(x_0*x_1)" or equations[0] == "y_0=(x_1*x_0)":
                        self.timesTested.value += 1

                    equationsNum = self.applySymbolic(path)

                    for item in equationsNum:
                        numTrivial += self.getTrivialOperations(item)

                    self.testedFunctions[equations[0]] = constantList[0]
                    return (outputs,flags,numTrivial)

                elif method == "evolutionary":
                    constants = self.fitConstantsEvolutionary(input,path,y)

    def trainFunction(self, dataGenerator, epochs, batchesPerEpoch, sampleSize, method, useMultiprocessing = False, saveState=False, saveStateSteps = None, trackHighestProb=False, plot=False, plotName = None):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        losses = []
        errors = []
        maxFunctionEpochs = []
        maxFunctionNames = []
        maxFunctionProbabilities = []

        stateEpochs = []
        state = []
        pathIndex = []
        pastPaths = []

        if useMultiprocessing:
            pool = multiprocessing.Pool(processes=14)


        numberCorrectEpochs = []
        numberCorrect = []

        startTime = time.perf_counter()
        self.converged = False
        for i in range(epochs):
            lossTotal = 0
            errorTotal = 0
            for j in range(batchesPerEpoch):
                pathIndex.append(i+j/batchesPerEpoch)

                numberCorrectEpochs.append(i*batchesPerEpoch+j)
                outputs = torch.empty((dataGenerator.batchSize,self.outputSize,self.recursiveDepth*sampleSize), dtype=torch.float)
                probabilities = torch.empty((sampleSize*self.recursiveDepth), dtype=torch.float)
                numTrivial = torch.empty((sampleSize*self.recursiveDepth))
                flags = torch.empty((sampleSize*self.recursiveDepth))
                numConstants = torch.empty((sampleSize*self.recursiveDepth))
                numActivations = torch.empty((sampleSize*self.recursiveDepth))

                x,y = dataGenerator.getBatch()

                paths,probs = self.getTrainingSamples(sampleSize)
                if saveState:
                    pastPaths.append(paths)

                index = 0
                if useMultiprocessing:
                    outputFlagsTrivial = pool.map(partial(self.forwardFitConstants, input = x, y = y, method = "gradient"),[[item[k] for item in paths] for k in range(sampleSize)])
                else:
                    inList = [[item[k] for item in paths] for k in range(sampleSize)]
                    outputFlagsTrivial = [self.forwardFitConstants(item,input = x, y = y, method = "gradient") for item in inList]

                for k in range(sampleSize):
                    outputs[:,:,index:index+self.recursiveDepth],flags[index:index+self.recursiveDepth],numTrivial[index:index+self.recursiveDepth] = outputFlagsTrivial[k]

                    probabilities[index:index+self.recursiveDepth] = probs[k]

                    numConstants[index:index+self.recursiveDepth] = self.getNumConstants([item[k] for item in paths])

                    numActivations[index:index+self.recursiveDepth] = self.getNumActivations([item[k] for item in paths])

                    index+=self.recursiveDepth

                lossVal,error = self.loss.getLossMultipleSamples(probabilities, y, outputs, numTrivial, flags, numConstants, numActivations)

                optimizer.zero_grad()
                lossVal.backward()
                optimizer.step()
                lossTotal += lossVal
                errorTotal += error

            if saveState:
                if i%saveStateSteps:
                    state.append([item.weight.clone() for item in self.layers])
                    stateEpochs.append(i)

            losses.append(lossTotal/batchesPerEpoch)
            errors.append(errorTotal/batchesPerEpoch)
            if len(errors) >= 30:
                conv = True
                for o in range(-2,-31,-1):
                    if not (abs(errors[-1]/errors[o]-1) < 0.01):
                        conv = False
                if conv == True:
                    self.converged = True
                    break

            if i%1 == 0:
                path,prob = self.getPathMaxProb()
                eqn = self.applySymbolicConstant(path)[0][0]
                print("Epoch "+str(i)+", Average Loss: "+str(losses[-1].item())+", Average Error: "+str(errors[-1].item())+", Best Function: "+self.applySymbolic(path)[0]+", With Probability: "+str(prob.item())+". Functions tested: "+str(len(self.testedFunctions)))

            if trackHighestProb:
                path,prob = self.getPathMaxProb()
                eqn = self.applySymbolicConstant(path)[0][0]
                if eqn in self.testedFunctions:
                    self.setConstants(self.testedFunctions[eqn])
                if maxFunctionNames == []:
                    maxFunctionNames.append(eqn)
                    maxFunctionEpochs.append([i])
                    maxFunctionProbabilities.append([prob])
                elif maxFunctionNames[-1] == eqn:
                    maxFunctionEpochs[-1].append(i)
                    maxFunctionProbabilities[-1].append(prob)
                else:
                    maxFunctionNames.append(eqn)
                    maxFunctionEpochs.append([i])
                    maxFunctionProbabilities.append([prob])
        
        if plot:
            with torch.no_grad():
                pathMaxProb,prob = self.getPathMaxProb()

                timesSampled = []
                correctEqn = self.applySymbolicConstant(pathMaxProb)[0][0]

                for paths in pastPaths:
                    inList = [[item[k] for item in paths] for k in range(sampleSize)]
                    numberSampled = 0
                    for path in inList:
                        eqn = self.applySymbolicConstant(path)[0][0]
                        if eqn == correctEqn:
                            numberSampled += 1
                    timesSampled.append(numberSampled)
                            

                bestPathProb = []
                for item in state:
                    for i in range(len(self.layers)):
                        self.layers[i].weight[:,:] = item[i]
                    bestPathProb.append(self.getProb(pathMaxProb))

                fig, ax1 = plt.subplots()

                for i in range(len(maxFunctionNames)):
                    ax1.plot(maxFunctionEpochs[i],maxFunctionProbabilities[i],label = maxFunctionNames[i])

                ax1.plot(stateEpochs, bestPathProb, label = self.applySymbolicConstant(pathMaxProb)[0][0])
                
                ax2 = ax1.twinx()
                ax2.plot(pathIndex,timesSampled)
                ax1.legend()
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Probability")
                ax2.set_ylabel("Times Sampled")
                ax1.set_yscale("log")
                plt.show()

                with open(f"{plotName}.dat","wb") as file:
                    pickle.dump((maxFunctionEpochs,maxFunctionProbabilities,maxFunctionNames,pathIndex,timesSampled,stateEpochs,bestPathProb, self.applySymbolicConstant(pathMaxProb)[0][0]), file)

    def getTimesSampled(self, maxProb, samples):
        timesSampled = []
        for sample in samples:
            matches = 0
            for i in range(sample[0].shape[0]):
                match = True
                for j in range(len(sample)):
                    if not torch.equal(maxProb[j],sample[j][i]):
                        match = False

                if match == True:
                    matches += 1
            timesSampled.append(matches)
        return timesSampled

            
    def trainFunctionMultiprocessing(self, dataGenerator, epochs, batchesPerEpoch, sampleSize, method):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        losses = []
        errors = []
        startTime = time.perf_counter()
        pool = multiprocessing.Pool(processes=14)
        self.converged = False
        for i in range(epochs):
            lossTotal = 0
            errorTotal = 0
            for j in range(batchesPerEpoch):
                outputs = torch.empty((dataGenerator.batchSize,self.outputSize,self.recursiveDepth*sampleSize), dtype=torch.float)
                probabilities = torch.empty((sampleSize*self.recursiveDepth), dtype=torch.float)
                numTrivial = torch.empty((sampleSize*self.recursiveDepth))
                flags = torch.empty((sampleSize*self.recursiveDepth))
                numConstants = torch.empty((sampleSize*self.recursiveDepth))
                numActivations = torch.empty((sampleSize*self.recursiveDepth))

                x,y = dataGenerator.getBatch()

                paths,probs = self.getTrainingSamples(sampleSize)
                index = 0
                outputFlagsTrivial = pool.map(partial(self.forwardFitConstants, input = x, y = y, method = "gradient"),[[item[k] for item in paths] for k in range(sampleSize)])
                for k in range(sampleSize):
                    outputs[:,:,index:index+self.recursiveDepth],flags[index:index+self.recursiveDepth],numTrivial[index:index+self.recursiveDepth] = outputFlagsTrivial[k]

                    probabilities[index:index+self.recursiveDepth] = probs[k]


                    numConstants[index:index+self.recursiveDepth] = self.getNumConstants([item[k] for item in paths])
  
                    numActivations[index:index+self.recursiveDepth] = self.getNumActivations([item[k] for item in paths])


                    index+=self.recursiveDepth

                lossVal,error = self.loss.getLossMultipleSamples(probabilities, y, outputs, numTrivial, flags, numConstants, numActivations)
                optimizer.zero_grad()
                lossVal.backward()
                optimizer.step()
                lossTotal += lossVal
                errorTotal += error


            losses.append(lossTotal/batchesPerEpoch)
            errors.append(errorTotal/batchesPerEpoch)
            if len(errors) >= 30:
                conv = True
                for o in range(-2,-31,-1):
                    if not (abs(errors[-1]/errors[o]-1) < 0.001):
                        conv = False
                if conv == True:
                    self.converged = True
                    break

            if i%10 == 0:
                path,prob = self.getPathMaxProb()
                eqn = self.applySymbolicConstant(path)[0][0]
                if eqn in self.testedFunctions:
                    self.setConstants(self.testedFunctions[eqn])
                print("Epoch "+str(i)+", Average Loss: "+str(losses[-1].item())+", Average Error: "+str(errors[-1].item())+", Best Function: "+self.applySymbolic(path)[0]+", With Probability: "+str(prob.item())+". Functions tested: "+str(len(self.testedFunctions)))