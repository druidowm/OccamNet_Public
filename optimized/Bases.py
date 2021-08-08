from abc import ABC,abstractmethod
import torch
import sympy as sp

class Base(ABC):
    @abstractmethod
    def getOutput(self, input):
        pass

    @abstractmethod
    def getSymbolicOutput(self, input):
        pass

class Add(Base):
    numInputs = 2

    def getLatex(self):
        return "+"

    def getOutput(self, input):
        return input[:,:,0]+input[:,:,1]

    def getOutputOne(self, input):
        return input[:,0]+input[:,1]

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"+"+str(input[1])+")"

    def copy(self):
        return Add()


class Subtract(Base):
    numInputs = 2

    def getLatex(self):
        return "-"

    def getOutput(self, input):
        return input[:,:,0]-input[:,:,1]

    def getOutputOne(self, input):
        return input[:,0]-input[:,1]

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"-"+str(input[1])+")"

    def copy(self):
        return Subtract()

class Multiply(Base):
    numInputs = 2

    def getLatex(self):
        return "x"

    def getOutput(self, input):
        return input[:,:,0]*input[:,:,1]

    def getOutputOne(self, input):
        y = input[:,0]*input[:,1]
        return y

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"*"+str(input[1])+")"

    def copy(self):
        return Multiply()

class Divide(Base):
    numInputs = 2

    def getLatex(self):
        return "÷"

    def getOutput(self, input):
        return input[:,:,0]/input[:,:,1]

    def getOutputOne(self, input):
        return input[:,0]/input[:,1]

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"/"+str(input[1])+")"

    def copy(self):
        return Divide()

class Sin(Base):
    numInputs = 1

    def getLatex(self):
        return "sin"

    def getOutput(self, input):
        return torch.sin(input[:,:,0])

    def getOutputOne(self, input):
        return torch.sin(input[:,0])

    def getSymbolicOutput(self, input):
        return "sin("+str(input[0])+")"

    def copy(self):
        return Sin()

class Cos(Base):
    numInputs = 1

    def getLatex(self):
        return "cos"

    def getOutput(self, input):
        return torch.cos(input[:,:,0])

    def getOutputOne(self, input):
        return torch.cos(input[:,0])

    def getSymbolicOutput(self, input):
        return "cos("+str(input[0])+")"

    def copy(self):
        return Cos()


class Exp(Base):
    numInputs = 1

    def getLatex(self):
        return "exp"

    def getOutput(self, input):
        return torch.exp(input[:,:,0])

    def getOutputOne(self, input):
        y = torch.exp(input[:,0])
        return y

    def getSymbolicOutput(self, input):
        return "exp("+str(input[0])+")"

    def copy(self):
        return Exp()


class Log(Base):
    numInputs = 1

    def getLatex(self):
        return "log"

    def getOutput(self, input):
        return torch.log(torch.abs(input[:,:,0]))

    def getOutputOne(self, input):
        return torch.log(torch.abs(input[:,0]))

    def getSymbolicOutput(self, input):
        return "log|"+str(input[0])+"|"

    def copy(self):
        return Log()
