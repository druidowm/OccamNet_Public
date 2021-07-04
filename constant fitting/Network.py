from numpy.lib.npyio import save
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import Bases
from DataGenerators import FunctionDataGenerator,ImplicitFunctionDataGenerator
from Losses import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR as decay
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

    def apply(self, input):
        output = torch.empty((input.shape[0],len(self.activations)))
        i = 0 
        for j in range(0,len(self.activations)):
            numInputs = self.activations[j].numInputs
            output[:,j] = self.activations[j].getOutput(input[:,i:i+numInputs])
            i+=numInputs

        return output

    def applySymbolic(self, input):
        output = []
        index = 0
        for item in self.activations:
            output.append(item.getSymbolicOutput(input[index:index+item.numInputs]))
            index += item.numInputs
        return output

    def applySymbolicConstant(self, input):
        output = []
        index = 0
        for item in self.activations:
            out = item.getSymbolicOutputConstant(input[index:index+item.numInputs])
            output.append(out)

            index += item.numInputs
        return output
    
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
    def __init__(self, inputSize, activationLists, outputSize, sparseSetter, loss, learningRate, constantLearningRate, temp, endTemp, equalization, recursiveDepth = 1, skipConnections = True):
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
            if equalization != 0:
                self.equalizeWeights()
                for layer in self.layers:
                    layer.weight[:,:] /= equalization

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

        for i in range(len(path)-1):

            inter = []

            for j in range(path[i].shape[0]):
                inter.append(input[path[i][j]])
            
            inter = self.activationLayers[i].applySymbolicConstant(inter)
            input = inter+input

        inter = []
        for j in range(path[-1].shape[0]):
            inter.append("y_"+str(j)+"="+input[path[-1][j]])

        return inter

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
            for j in range(len(self.layers)-1):
                img = torch.gather(input, 1, path[j])

                inter = self.activationLayers[j].apply(img)
                input = torch.cat([inter,input], dim = 1)

            input = torch.gather(input, 1, path[-1])

            outputs[:,:,i] = input
        return outputs

    def fitConstantsGradient(self, input, path, y):
        MSELoss = nn.MSELoss()
        constantList = torch.empty((self.recursiveDepth,self.totalConstants), dtype = torch.float)
        outputs = torch.empty((input.shape[0],y.shape[1],self.recursiveDepth), dtype = torch.float)

        for j in range(self.recursiveDepth):
            self.constants = nn.Parameter(torch.rand([self.totalConstants], dtype=torch.float))
            optimizer = torch.optim.Adam(self.parameters(), lr=self.constantLearningRate)
            self.setConstants(self.constants)

            output = self.forward(input,path)
            output = output[:,:,j]

            count = 0
            while torch.any(output!=output) and count<100:
                self.constants = nn.Parameter(torch.rand([self.totalConstants], dtype=torch.float))
                self.setConstants(self.constants)
                output = self.forward(input,path)
                output = output[:,:,j]
                count+=1
            
            if count >= 100:
                outputs[:,:,j] = output
                constantList[j,:] = self.constants
                break

            for i in range(2000):
                lossVal = MSELoss(y,output)

                optimizer.zero_grad()
                lossVal.backward()
        
                if torch.all(torch.abs(self.constants.grad)<0.0001) or torch.any(self.constants.grad!=self.constants.grad):
                    break

                optimizer.step()
                output = self.forward(input,path)
                output = output[:,:,j]

                if torch.any(output!=output):
                    break

            outputs[:,:,j] = output
            constantList[j,:] = self.constants

        return (constantList.detach(),outputs.detach())

    def fitConstantsEvolutionary(self, input, path, y):
        pass

    def forwardFitConstants(self, path, input = None, y = None, method = "gradient"):
        if self.constants.shape[0]==0:
            return self.forward(input,path)

        else:
            equations = self.applySymbolicConstant(path)
            if equations[0] in self.testedFunctions:
                self.setConstants(self.testedFunctions[equations[0]])
                return self.forward(input,path)
                
            else:
                constantList,outputs = self.fitConstantsGradient(input,[item.detach() for item in path],y)
                equations = self.applySymbolicConstant(path)
                self.testedFunctions[equations[0]] = constantList[0]
                return outputs

    def trainFunction(self, epochs, batchesPerEpoch, sampleSize, decayRate, train_X, train_Y, useMultiprocessing = False, numProcesses = 2, trackHighestProb=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        scheduler = decay(optimizer, decayRate)
        losses = []
        errors = []

        maxFunctionEpochs = []
        maxFunctionNames = []
        maxFunctionProbabilities = []


        pathIndex = []

        if useMultiprocessing:
            pool = multiprocessing.Pool(processes=numProcesses)

        numberCorrectEpochs = []

        self.converged = False
        for i in range(epochs):
            lossTotal = 0
            errorTotal = 0
            for j in range(batchesPerEpoch):
                pathIndex.append(i+j/batchesPerEpoch)

                numberCorrectEpochs.append(i*batchesPerEpoch+j)
                outputs = torch.empty((train_X.shape[0],self.outputSize,self.recursiveDepth*sampleSize), dtype=torch.float)
                probabilities = torch.empty((sampleSize*self.recursiveDepth), dtype=torch.float)

                paths,probs = self.getTrainingSamples(sampleSize)

                index = 0
                if useMultiprocessing:
                    outputList = pool.map(partial(self.forwardFitConstants, input = train_X, y = train_Y),[[item[k] for item in paths] for k in range(sampleSize)])
                else:
                    inList = [[item[k] for item in paths] for k in range(sampleSize)]
                    outputList = [self.forwardFitConstants(item,input = train_X, y = train_Y) for item in inList]

                for k in range(sampleSize):
                    outputs[:,:,index:index+self.recursiveDepth] = outputList[k]
                    probabilities[index:index+self.recursiveDepth] = probs[k]

                    index+=self.recursiveDepth

                lossVal,error = self.loss.getLossMultipleSamples(probabilities, train_Y, outputs)

                optimizer.zero_grad()
                lossVal.backward()
                optimizer.step()

                scheduler.step()

                lossTotal += lossVal
                errorTotal += error
                losses.append(lossTotal/batchesPerEpoch)
                errors.append(errorTotal/batchesPerEpoch)

            if i%1 == 0:
                path,prob = self.getPathMaxProb()
                eqn = self.applySymbolicConstant(path)[0]
                if eqn in self.testedFunctions:
                    self.setConstants(self.testedFunctions[eqn])
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