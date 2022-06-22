import numpy as np
import math

import Bases
from ActivationLayer import ActivationLayer

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib import patches as patch

from sklearn.metrics import mean_squared_error as MSE

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR as decay

import multiprocessing
from multiprocessing import Value
from functools import partial

import datetime
import time

class Network(nn.Module):
    def __init__(self, inputSize, activationLists, outputSize, sparseSetter, loss, learningRate, temp, endTemp, equalization):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

        self.inputSize = inputSize
        self.activationLayers = sparseSetter.getActivationsSparsity(inputSize,activationLists,outputSize)
        self.outputSize = outputSize
        self.temp = temp
        self.endTemp = endTemp
        self.totalConstants = 0
        self.loss = loss
        self.learningRate = learningRate
        self.layers = [nn.Linear(inputSize,self.activationLayers[0].totalInputs,bias=False).to(self.device)]

        prevLayerSize = self.inputSize
        for i in range(0,len(self.activationLayers)-1):
            prevLayerSize+=len(self.activationLayers[i].activations)
            self.layers.append(nn.Linear(prevLayerSize,self.activationLayers[i+1].totalInputs, bias = False).to(self.device))
        self.layers.append(nn.Linear(len(self.activationLayers[-1].activations)+prevLayerSize,outputSize, bias = False).to(self.device))

        self.layers = nn.ModuleList(self.layers)

        #self.plot()

        with torch.no_grad():
            for layer in self.layers:
                layer.weight[:,:] = 0
            if equalization != 0:
                self.equalizeWeights()
                for layer in self.layers:
                    layer.weight[:,:] /= equalization

    def equalizeWeights(self):
        weight = F.softmax(self.layers[0].weight/self.temp, dim=1)
        prob2 = torch.tensor([torch.max(weight[i]) for i in range(self.activationLayers[0].totalInputs)], dtype=torch.double).to(self.device)


        for i in range(0,len(self.layers)-1):
            prob3 = torch.ones([len(self.activationLayers[i].activations)],dtype = torch.double).to(self.device)
            index = 0
            for j in range(0,prob3.shape[0]):
                for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                    prob3[j] *= prob2[k]
                index+= self.activationLayers[i].activations[j].numInputs

            if i == 0:
                prob = torch.cat([prob3,torch.ones([self.inputSize], dtype = torch.double).to(self.device)], dim = 0)
            else:
                prob = torch.cat([prob3,prob], dim = 0)
            
            numProbs = []
            probWeights = []
            for j in range(prob.shape[0]):
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
                matLayer = [0 for k in range(len(probWeights))]
                matLayer[0] = probWeights[0]
                matLayer[j] = -probWeights[j]
                probMatrix.append(matLayer)

            probMatrix = torch.tensor(probMatrix, dtype = torch.double).to(self.device)

            outVec = [0 for k in range(len(probWeights))]
            outVec[0] = 1
            outVec = torch.tensor(outVec,dtype = torch.double).to(self.device)

            weightVals = torch.matmul(probMatrix.inverse(),outVec)


            if i == len(self.layers)-2:
                bot = 1/weightVals[0]
                weightVals = self.endTemp*torch.log(bot*weightVals)
            else:
                bot = 1/weightVals[0]
                weightVals = self.temp*torch.log(bot*weightVals)

            for k in range(prob.shape[0]):
                for j in range(0,len(probWeights)):
                    if abs(prob[k]/probWeights[j]-1)<0.01:
                        self.layers[i+1].weight[:,k] = weightVals[j]


            if i == len(self.layers)-2:
                weight = F.softmax(self.layers[i+1].weight/self.endTemp, dim=1)
            else:
                weight = F.softmax(self.layers[i+1].weight/self.temp, dim=1)

            prob2 = prob[0]*weight[:,0]


    def plotNode(self,pos,text = None):
        if text != None:
            plt.text(pos[0],pos[1],text,fontsize = 5,horizontalalignment = "center",verticalalignment = "center",zorder=3)
        theta = np.arange(0,2*math.pi+2*math.pi/300,2*math.pi/300)
        plt.fill(pos[0]+15*np.cos(theta),pos[1]+15*np.sin(theta),"r",zorder=2)

    def plotLayer(self,prevXpos,xpos,numNodes,numPrevNodes):
        for i in range(numNodes):
            ypos = 100*i
            for k in range(numPrevNodes):
                plt.plot([prevXpos, xpos],[100*k,ypos],"g",zorder = 1)


            self.plotNode((xpos,ypos))

    def plotActivationLayer(self,prevXpos,xpos,activations,numSkipNodes):
        index = 0
        for i in range(len(activations)):
            ypos = 100*i
            xdelta = -100
            for j in range(activations[i].numInputs):
                ydelta = 40*j-20*activations[i].numInputs
                plt.plot([xpos+xdelta,xpos],[ypos+ydelta,ypos],"b",zorder = 1)
                for k in range(numSkipNodes):
                    plt.plot([prevXpos, xpos+xdelta],[100*k,ypos+ydelta],"g",zorder = 1)

                self.plotNode((xpos+xdelta,ypos+ydelta))
                index += 1


            self.plotNode((xpos,ypos),activations[i].getLatex())

        for i in range(numSkipNodes):
            ypos = 100*(i+len(activations))
            plt.plot([prevXpos,xpos],[100*i,ypos],"k--",zorder = 1)
            self.plotNode((xpos,ypos))


    def plot(self):
        self.plotNode((0,0))
        numSkip = self.inputSize
        for i in range(len(self.activationLayers)):
            xpos = i*500+500
            self.plotActivationLayer(xpos-500,xpos,self.activationLayers[i].activations,numSkip)
            numSkip += len(self.activationLayers[i].activations)

        self.plotLayer(500*len(self.activationLayers),500*len(self.activationLayers)+500,self.outputSize,numSkip)
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

    def getTrainingSamples(self, sampleSize):
        paths = []
        logprobs = torch.zeros((sampleSize,self.inputSize), dtype = torch.float, device = self.device)

        for i in range(0,len(self.layers)-1):
            weight = F.softmax(self.layers[i].weight/self.temp, dim=1)

            path = Categorical(weight).sample([sampleSize])

            paths.append(path)

            logprobs2 = torch.gather(logprobs, 1, path) + torch.log(torch.gather(weight.T, 0, path))

            logprob = torch.zeros((sampleSize,len(self.activationLayers[i].activations)), device = self.device)
            index = 0
            for j in range(0,logprob.shape[1]):
                for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                    logprob[:,j] += logprobs2[:,k]
                index+= self.activationLayers[i].activations[j].numInputs


            logprobs = torch.cat([logprob,logprobs], dim = 1)


        weight = F.softmax(self.layers[-1].weight/self.endTemp, dim=1)

        try:
            path = Categorical(weight).sample([sampleSize])
        except Exception as e:
            print(e)
            print(f"weight is {weight}")
            print(f"layer weights are {self.layers[-1].weight}")
            print(f"scaled layer weights are {self.layers[-1].weight}")
            raise e

        paths.append(path)


        logprobs = torch.gather(logprobs, 1, path) + torch.log(torch.gather(weight.T, 0, path))
        logprob = torch.sum(logprobs,1)

        return (paths,logprob)


    def getTrainingSamples2(self, sampleSize):
        paths = []
        logprobs = torch.zeros((sampleSize,self.inputSize), dtype = torch.float, device = self.device)

        for i in range(0,len(self.layers)-1):
            weight = F.softmax(self.layers[i].weight/self.temp, dim=1)
        
            path = Categorical(logits = self.layers[i].weight/self.temp).sample([sampleSize])

            paths.append(path)

            logprobs2 = torch.gather(logprobs, 1, path) + torch.log(torch.gather(weight.T, 0, path))

            logprob = torch.zeros((sampleSize,len(self.activationLayers[i].activations)), device = self.device)
            index = 0
            for j in range(0,logprob.shape[1]):
                for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                    logprob[:,j] += logprobs2[:,k]
                index+= self.activationLayers[i].activations[j].numInputs


            logprobs = torch.cat([logprob,logprobs], dim = 1)


        weight = F.softmax(self.layers[-1].weight/self.endTemp, dim=1)

        path = Categorical(logits = self.layers[-1].weight/self.endTemp).sample([sampleSize])
        paths.append(path)


        logprobs = torch.gather(logprobs, 1, path) + torch.log(torch.gather(weight.T, 0, path))
        logprob = torch.sum(logprobs,1)

        return (paths,logprob)


    def getPathMaxProb(self):
        with torch.no_grad():
            path = []
            prob = torch.ones([self.inputSize], dtype = torch.float, device = self.device)

            for i in range(0,len(self.layers)-1):
                weight = F.softmax(self.layers[i].weight/self.temp, dim=1)
                splProbs = prob.unsqueeze(0).repeat((weight.shape[0],1))
                splProbs = weight*splProbs
                path.append(torch.argmax(splProbs, dim = 1))


                probs = torch.gather(splProbs.T, 0, path[i].unsqueeze(0))[0]

                prob2 = torch.ones([len(self.activationLayers[i].activations)], device = self.device)
                index = 0
                for j in range(0,prob2.shape[0]):
                    for k in range(index,index+self.activationLayers[i].activations[j].numInputs):
                        prob2[j] *= probs[k]
                    index+= self.activationLayers[i].activations[j].numInputs


                prob = torch.cat([prob2,prob], dim = 0)


            weight = F.softmax(self.layers[-1].weight/self.endTemp, dim=1)
            splProbs = prob.unsqueeze(0).repeat((weight.shape[0],1))
            splProbs = weight*splProbs
            path.append(torch.argmax(splProbs, dim = 1))

            probs = torch.gather(splProbs.T, 0, path[-1].unsqueeze(0))[0]
            prob = torch.prod(probs)
            return (path,prob)

    def forward(self, input, paths):
        with torch.no_grad():
            input = input.unsqueeze(1).repeat(1, paths[0].shape[0], 1)
            for j in range(len(self.layers)-1):
                img = torch.gather(input, 2, paths[j].unsqueeze(0).repeat(input.shape[0],1,1))

                inter = self.activationLayers[j].apply(img)

                input = torch.cat([inter,input], dim = 2)

            input = torch.gather(input, 2, paths[-1].unsqueeze(0).repeat(input.shape[0],1,1))

            return input

    def forwardOneFunction(self, input, path):
        with torch.no_grad():
            for j in range(len(self.layers)-1):
                img = input[:,path[j]]

                inter = self.activationLayers[j].applyOne(img)

                input = torch.cat([inter,input], dim = 1)

            input = input[:,path[-1]]

            return input

    def trainFunction(self, epochs, sampleSize, decayRate, train_X, train_Y, val_X = None, val_Y = None):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        scheduler = decay(optimizer, decayRate)
        losses = []
        errors = []

        MSELoss = nn.MSELoss()

        bestTrainFunction = None
        bestTrainFunctionError = float("inf")

        bestValFunction = None
        bestValFunctionError = float("inf")

        train_Y_unsqueeze = train_Y.unsqueeze(1)

        for i in range(epochs):
            paths,logprobs = self.getTrainingSamples(sampleSize)

            output = self.forward(train_X, paths)

            lossVal = self.loss.getLossMultipleSamples(logprobs, train_Y_unsqueeze, output)

            optimizer.zero_grad()
            lossVal.backward()
            optimizer.step()

            scheduler.step()

            losses.append(lossVal.item())

            path,prob = self.getPathMaxProb()
            if i%100 == 0:
                print("Epoch "+str(i)+", Average Loss: "+str(losses[-1])+", Best Function: "+self.applySymbolic(path)[0]+", With Probability: "+str(prob.item()))


            pred = self.forwardOneFunction(train_X, path)[:,0]
            bestPathError = MSELoss(train_Y,pred).item()

            if bestPathError < bestTrainFunctionError:
                bestTrainFunctionError = bestPathError
                bestTrainFunction = [item.cpu().numpy() for item in path]

            if val_X != None:
                pred = self.forwardOneFunction(val_X, path)[:,0]
                bestPathError = MSELoss(val_Y,pred).item()

                if bestPathError < bestValFunctionError:
                    bestValFunctionError = bestPathError
                    bestValFunction = [item.cpu().numpy() for item in path]

        return (bestTrainFunction, bestValFunction)

    def trainFunctionBatch(self, epochs, sampleSize, batchSize, decayRate, train_X, train_Y, val_X = None, val_Y = None):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        scheduler = decay(optimizer, decayRate)
        losses = []
        errors = []

        MSELoss = nn.MSELoss()

        bestTrainFunction = None
        bestTrainFunctionError = float("inf")

        bestValFunction = None
        bestValFunctionError = float("inf")

        train_Y_unsqueeze = train_Y.unsqueeze(1)

        for i in range(epochs):
            logprobs = torch.empty((sampleSize*batchSize,), dtype=float, device = self.device)
            output = torch.empty((train_X.shape[0],sampleSize*batchSize,1), dtype=float, device = self.device)
            
            for j in range(batchSize):
                paths,logprobs[j*sampleSize:(j+1)*sampleSize] = self.getTrainingSamples2(sampleSize)
                output[:,j*sampleSize:(j+1)*sampleSize,:] = self.forward(train_X, paths)

            lossVal = self.loss.getLossMultipleSamples(logprobs, train_Y_unsqueeze, output)

            optimizer.zero_grad()
            lossVal.backward()
            optimizer.step()

            scheduler.step()

            losses.append(lossVal.item())

            path,prob = self.getPathMaxProb()
            if i%100 == 0:
                print("Epoch "+str(i)+", Average Loss: "+str(losses[-1])+", Best Function: "+self.applySymbolic(path)[0]+", With Probability: "+str(prob.item()))


            pred = self.forwardOneFunction(train_X, path)[:,0]
            bestPathError = MSELoss(train_Y,pred).item()

            if bestPathError < bestTrainFunctionError:
                bestTrainFunctionError = bestPathError
                bestTrainFunction = [item.cpu().numpy() for item in path]

            if val_X != None:
                pred = self.forwardOneFunction(val_X, path)[:,0]
                bestPathError = MSELoss(val_Y,pred).item()

                if bestPathError < bestValFunctionError:
                    bestValFunctionError = bestPathError
                    bestValFunction = [item.cpu().numpy() for item in path]

        return (bestTrainFunction, bestValFunction)
