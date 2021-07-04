import torch
import math
from NetworkRegularization import ActivationLayer

class SetPartialSparse:
    def __init__(self, sparseInputs):
        self.sparseInputs = sparseInputs

    def getActivationsSparsity(self, inputSize, activationLists, outputSize):
        numItems = [outputSize]
        for i in range(len(activationLists)-2,-1,-1):
            numItems.insert(0,numItems[0]*self.getMaxInputs(activationLists[i]))

        newActivationLists = []

        prevLayer = inputSize
        for k in range(0,len(activationLists)):
            newActivationList = []
            for i in range(len(activationLists[k])):
                sparse = False
                for item in self.sparseInputs:
                    if isinstance(activationLists[k][i],item):
                        sparse = True
                        for j in range(numItems[k]*(prevLayer**activationLists[k][i].numInputs)):  
                            newActivationList.append(activationLists[k][i].copy())
                
                if not sparse:
                    for j in range(numItems[k]):
                        newActivationList.append(activationLists[k][i])

            newActivationLists.append(newActivationList)
            prevLayer += len(newActivationLists[-1])

        for i in range(0,len(newActivationLists)):
            newActivationLists[i] = ActivationLayer(newActivationLists[i])

        sparse = []
        prevLayer = inputSize
        for i in range(len(newActivationLists)):
            sparseLayer = torch.ones([newActivationLists[i].totalInputs,prevLayer],dtype = torch.bool)
            outIndex = 0

            inIndexes = [0 for k in range(newActivationLists[i].activations[j].numInputs)]
            for j in range(len(newActivationLists[i].activations)):
                for item in self.sparseInputs:
                    if isinstance(newActivationLists[i].activations[j],item):
                        sparseLayer[outIndex:outIndex+newActivationLists[i].activations[j].numInputs,:] = False
                        for k in range(newActivationLists[i].activations[j].numInputs):
                            sparseLayer[outIndex,inIndexes[k]] = True
                            outIndex += 1
                        
                        inIndexes[-1]+=1
                        add = False
                        for l in range(len(inIndexes)-1,-1,-1):
                            if add:
                                inIndexes[l]+=1
                            if inIndexes[l]>=prevLayer:
                                inIndexes[l]=0
                                add = True
                            else:
                                add = False
                    else:
                        outIndex+=newActivationLists[i].activations[j].numInputs

            sparse.append(sparseLayer)
            prevLayer += len(newActivationLists[i].activations)

        sparseLayer = torch.ones([outputSize,prevLayer],dtype = torch.bool)

        sparse.append(sparseLayer)


        return (newActivationLists,sparse)



    def getMaxInputs(self, activationList):
        maxInputs = 0
        for item in activationList:
            if item.numInputs > maxInputs:
                maxInputs = item.numInputs
        
        return maxInputs


class SetNoSparse:
    def getActivationsSparsity(self, inputSize, activationLists, outputSize):
        numItems = [outputSize]
        for i in range(len(activationLists)-2,-1,-1):
            numItems.insert(0,numItems[0]*self.getMaxInputs(activationLists[i]))

        newActivationLists = []

        prevLayer = inputSize
        for k in range(0,len(activationLists)):
            newActivationList = []
            for i in range(len(activationLists[k])):
                for j in range(numItems[k]):
                    newActivationList.append(activationLists[k][i])

            newActivationLists.append(newActivationList)
            prevLayer += len(newActivationLists[-1])

        for i in range(0,len(newActivationLists)):
            newActivationLists[i] = ActivationLayer(newActivationLists[i])

        sparse = []
        prevLayer = inputSize
        for i in range(len(newActivationLists)):
            sparseLayer = torch.ones([newActivationLists[i].totalInputs,prevLayer],dtype = torch.bool)
            sparse.append(sparseLayer)
            prevLayer += len(newActivationLists[i].activations)

        sparseLayer = torch.ones([outputSize,prevLayer],dtype = torch.bool)

        sparse.append(sparseLayer)


        return (newActivationLists,sparse)



    def getMaxInputs(self, activationList):
        maxInputs = 0
        for item in activationList:
            if item.numInputs > maxInputs:
                maxInputs = item.numInputs
        
        return maxInputs

class SetNoSparseNoDuplicates:
    def getActivationsSparsity(self, inputSize, activationLists, outputSize):
        numItems = [1 for i in range(len(activationLists))]

        newActivationLists = []

        prevLayer = inputSize
        for k in range(0,len(activationLists)):
            newActivationList = []
            for i in range(len(activationLists[k])):
                for j in range(numItems[k]):
                    newActivationList.append(activationLists[k][i])

            newActivationLists.append(newActivationList)
            prevLayer += len(newActivationLists[-1])

        for i in range(0,len(newActivationLists)):
            newActivationLists[i] = ActivationLayer(newActivationLists[i])

        sparse = []
        prevLayer = inputSize
        for i in range(len(newActivationLists)):
            sparseLayer = torch.ones([newActivationLists[i].totalInputs,prevLayer],dtype = torch.bool)
            sparse.append(sparseLayer)
            prevLayer += len(newActivationLists[i].activations)

        sparseLayer = torch.ones([outputSize,prevLayer],dtype = torch.bool)

        sparse.append(sparseLayer)


        return (newActivationLists,sparse)



    def getMaxInputs(self, activationList):
        maxInputs = 0
        for item in activationList:
            if item.numInputs > maxInputs:
                maxInputs = item.numInputs
        
        return maxInputs
