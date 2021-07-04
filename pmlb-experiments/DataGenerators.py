import torch

class FunctionDataGenerator:
    def __init__(self, batchSize, dataRange, function):
        self.batchSize = batchSize
        self.dataRange = dataRange
        self.function = function

    def getBatch(self):
        x = (torch.rand([self.batchSize], dtype = torch.float)*(self.dataRange[1]-self.dataRange[0])+self.dataRange[0]).unsqueeze(1)
        y = self.function(x)
        return (x,y)

class MultivariateFunctionDataGenerator:
    def __init__(self, batchSize, dataRanges, function, numInputs):
        self.batchSize = batchSize
        self.dataRanges = dataRanges
        self.function = function
        self.numInputs = numInputs

    def getBatch(self,j):
        data = []
        for i in range(self.numInputs):
            data.append((torch.rand([self.batchSize], dtype = torch.float)*(self.dataRanges[i][1]-self.dataRanges[i][0])+self.dataRanges[i][0]).unsqueeze(1))

        data = torch.cat(data,dim = 1)
        y = self.function(data).unsqueeze(1)
        return (data,y)

    def updateBatches(self):
        pass

class ImplicitFunctionDataGenerator:
    def __init__(self, numInputs, batchSize, dataRange, function, output):
        self.batchSize = batchSize
        self.dataRange = dataRange
        self.numInputs = numInputs
        self.function = function
        self.output = output

    def getBatch(self):
        data = []
        for i in range(self.numInputs-1):
            data.append((torch.rand([self.batchSize], dtype = torch.float)*(self.dataRange[1]-self.dataRange[0])+self.dataRange[0]).unsqueeze(1))
        data.append(self.function(data))

        y = torch.tensor([[self.output] for i in range(self.batchSize)])
        return (torch.cat(data,dim = 1),y)

class ImplicitFunctionDataGeneratorSample:
    def __init__(self, numInputs, batchSize, dataRange, sampleSize, function, output, random = True):
        self.batchSize = batchSize
        self.dataRange = dataRange
        self.numInputs = numInputs
        self.sampleSize = sampleSize
        self.function = function
        self.output = output

        self.inputs = [torch.rand([self.sampleSize,1], dtype = torch.float)*(self.dataRange[1]-self.dataRange[0])+self.dataRange[0] for i in range(numInputs-1)]
        self.inputs.append(self.function(self.inputs))
        print(self.inputs)

    def getBatch(self):
        data = []
        inputI = torch.rand([self.batchSize,1], dtype = torch.float)
        inputI = torch.floor(self.sampleSize*inputI).type(torch.long)

        for i in range(self.numInputs):
            data.append(torch.gather(self.inputs[i],0,inputI))

        y = torch.tensor([[self.output] for i in range(self.batchSize)])
        return (torch.cat(data,dim = 1),y)


class FunctionDataGeneratorSample:
    def __init__(self, batchSize, sampleSize, dataRange, function, constants = None):
        self.batchSize = batchSize
        self.dataRange = dataRange
        self.function = function
        self.sampleSize = sampleSize

        self.input = torch.rand([self.sampleSize,1], dtype = torch.float)*(self.dataRange[1]-self.dataRange[0])+self.dataRange[0]
        self.output = self.function(self.input)
        if constants != None:
            for item in constants:
                addOn = torch.ones([self.sampleSize,1])*item
                self.input = torch.cat([self.input,addOn],dim = 1)

    def getBatch(self,i):
        inputI = torch.rand([self.batchSize,1], dtype = torch.float)
        inputI = torch.floor(self.sampleSize*inputI).type(torch.long)
        y = torch.gather(self.output,0,inputI)
        inputI = inputI.repeat(1,self.input.shape[1])
        x = torch.gather(self.input,0,inputI)
        return (x,y)

    def updateBatches(self):
        pass


class DataGeneratorSample:
    def __init__(self, batchSize, train_X, train_Y):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batchSize = batchSize
        self.train_X = train_X
        self.train_Y = train_Y.unsqueeze(1)
        self.perm = torch.randperm(self.train_X.shape[0])

    def updateBatches(self):
        self.perm = torch.randperm(self.train_X.shape[0])

    def getBatch(self, i):
        indices = self.perm[i*self.batchSize:(i+1)*self.batchSize]
        print(indices)
        x = self.train_X[indices]
        y = self.train_Y[indices]
        return (x,y)
