import torch

class ActivationLayer:
    def __init__(self, activations):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.activations = activations
        self.totalInputs = 0
        for item in activations:
            self.totalInputs += item.numInputs

    def apply(self, input):
        output = torch.empty((input.shape[0],input.shape[1],len(self.activations)), device = self.device)
        i = 0
        for j in range(0,len(self.activations)):
            numInputs = self.activations[j].numInputs
            output[:,:,j]= self.activations[j].getOutput(input[:,:,i:i+numInputs])

            i+=numInputs
        return output

    def applyOne(self, input):
        output = torch.empty((input.shape[0],len(self.activations)), device = self.device)
        i = 0
        for j in range(0,len(self.activations)):
            numInputs = self.activations[j].numInputs
            output[:,j]= self.activations[j].getOutputOne(input[:,i:i+numInputs])

            i+=numInputs
        return output

    def applySymbolic(self, input):
        output = []
        index = 0
        for item in self.activations:
            output.append(item.getSymbolicOutput(input[index:index+item.numInputs]))
            index += item.numInputs
        return output
