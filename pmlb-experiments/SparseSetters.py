import torch
import math
from ActivationLayer import ActivationLayer

class SetNoSparse:
    def getActivationsSparsity(self, inputSize, activationLists, outputSize):
        numItems = [outputSize]
        for i in range(len(activationLists)-1,0,-1):
            numItems.insert(0,numItems[0]*self.getMaxInputs(activationLists[i]))

        newActivationLists = []

        for k in range(0,len(activationLists)):
            newActivationList = []
            for i in range(len(activationLists[k])):
                for j in range(numItems[k]):
                    newActivationList.append(activationLists[k][i])

            newActivationLists.append(newActivationList)

        for i in range(0,len(newActivationLists)):
            newActivationLists[i] = ActivationLayer(newActivationLists[i])

        return newActivationLists



    def getMaxInputs(self, activationList):
        maxInputs = 0
        for item in activationList:
            if item.numInputs > maxInputs:
                maxInputs = item.numInputs

        return maxInputs
