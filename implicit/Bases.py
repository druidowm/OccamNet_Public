from abc import ABC,abstractmethod
import torch
import sympy as sp
import numpy as np

#Nan represents unfixed units. Not wrong units.
def checkNan(input):
    return np.isnan(input[0])

#Inf represents wrong units that need to be propagated further
def checkInf(input):
    return np.isinf(input[0])

def matchUnits(unit1,unit2):
    if np.any(unit1 != unit2):
        if checkNan(unit1):
            return unit2
        if checkNan(unit2):
            return unit1
        return np.full(unit1.shape,np.inf)
    return unit1

def checkNonzero(unit):
    if np.any(unit):
        if not checkNan(unit):
            return np.full(unit.shape,np.inf)
        return np.zeros(unit.shape)
    return unit

class Base(ABC):
    @abstractmethod
    def getOutput(self, input):
        pass

    @abstractmethod
    def getSymbolicOutput(self, input):
        pass
    
    @abstractmethod
    def propagateUnits(self, input):
        pass

class BaseWithConstants(Base):
    numInputs = 1
    def getConstant(self, constant):
        self.constant = constant
    
    def getSymbolicConstant(self,constant):
        self.symbolicConstant = constant

class Add(Base):
    numInputs = 2

    def getLatex(self):
        return "+"

    def getOutput(self, input):
        return (input[:,0]+input[:,1],0)
    
    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"+"+str(input[1])+")"
    
    def getSymbolicOutputConstant(self, input):
        return ("("+str(input[0])+"+"+str(input[1])+")", 0)

    def copy(self):
        return Add()

    def propagateUnits(self, input):
        return matchUnits(input[0],input[1])


class Add3(Base):
    numInputs = 3

    def getLatex(self):
        return "+"

    def getOutput(self, input):
        return (input[:,0]+input[:,1]+input[:,2],0)
    
    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"+"+str(input[1])+"+"+str(input[2])+")"
    
    def getSymbolicOutputConstant(self, input):
        return ("("+str(input[0])+"+"+str(input[1])+"+"+str(input[2])+")", 0)

    def copy(self):
        return Add3()

    def propagateUnits(self, input):
        input1 = matchUnits(input[0],input[1])
        return matchUnits(input[2],input1)

class Subtract(Base):
    numInputs = 2

    def getLatex(self):
        return "-"

    def getOutput(self, input):
        return (input[:,0]-input[:,1],0)

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"-"+str(input[1])+")"

    def getSymbolicOutputConstant(self, input):
        return ("("+str(input[0])+"-"+str(input[1])+")",0)

    def copy(self):
        return Subtract()

    def propagateUnits(self, input):
        return matchUnits(input[0],input[1])

class Multiply(Base):
    numInputs = 2

    def getLatex(self):
        return "×"

    def getOutput(self, input):
        return (input[:,0]*input[:,1],0)

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"*"+str(input[1])+")"

    def getSymbolicOutputConstant(self, input):
        return ("("+str(input[0])+"*"+str(input[1])+")",0)
    
    def copy(self):
        return Multiply()
    
    def propagateUnits(self, input):
        if checkInf(input[0]):
            return input[0]
        if checkInf(input[1]):
            return input[1]
        return input[0]+input[1]

class Divide(Base):
    numInputs = 2

    def getLatex(self):
        return "÷"

    def getOutput(self, input):
        return (input[:,0]/input[:,1],0)

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"/"+str(input[1])+")"

    def getSymbolicOutputConstant(self, input):
        return ("("+str(input[0])+"/"+str(input[1])+")",0)

    def copy(self):
        return Divide()

    def propagateUnits(self, input):
        if checkInf(input[0]):
            return input[0]
        if checkInf(input[1]):
            return input[1]

        return input[0]-input[1]

class Sin(Base):
    numInputs = 1

    def getLatex(self):
        return "cos"

    def getFlags(self, inp, output):
        if torch.mean(1-output)<0.25:
            return min(5,0.1/torch.mean(1-output).detach())
        if torch.mean(output+1)<0.25:
            return min(5,0.1/torch.mean(output+1).detach())
        if torch.mean(torch.abs(output-inp))<0.1:
            return min(5,0.05/torch.mean(torch.abs(output-inp)).detach())
        return 0

    def getOutput(self, input):
        output = torch.sin(input[:,0])
        return (output,self.getFlags(input[:,0],output))

    def getSymbolicOutput(self, input):
        return "sin("+str(input[0])+")"

    def getSymbolicOutputConstant(self, input):
        return ("sin("+str(input[0])+")",0)

    def copy(self):
        return Sin()

    def propagateUnits(self, input):
        return checkNonzero(input[0])

class Cos(Base):
    numInputs = 1

    def getLatex(self):
        return "cos"

    def getFlags(self, output):
        if torch.mean(1-output)<0.25:
            return min(5,0.1/torch.mean(1-output).detach())
        if torch.mean(output+1)<0.25:
            return min(5,0.1/torch.mean(output+1).detach())
        return 0

    def getOutput(self, input):
        output = torch.cos(input[:,0])
        return (output,self.getFlags(output))

    def getSymbolicOutput(self, input):
        return "cos("+str(input[0])+")"

    def getSymbolicOutputConstant(self, input):
        return ("cos("+str(input[0])+")",0)

    def copy(self):
        return Cos()

    def propagateUnits(self, input):
        return checkNonzero(input[0])


class Square(Base):
    numInputs = 1

    def getLatex(self):
        return "x^2"

    def getFlags(self, output):
        if torch.mean(output)<0.25:
            return min(5,0.01/torch.mean(output).detach())
        return 0

    def getOutput(self, input):
        output = input[:,0]*input[:,0]
        return (output,self.getFlags(output))

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"^2)"

    def getSymbolicOutputConstant(self, input):
        return ("("+str(input[0])+"^2)",0)

    def copy(self):
        return Square()

    def propagateUnits(self, input):
        return 2*input[0]

class Cube(Base):
    numInputs = 1

    def getLatex(self):
        return "x^3"

    def getFlags(self, output):
        if torch.mean(torch.abs(output))<0.25:
            return min(5,0.01/torch.mean(torch.abs(output)).detach())
        return 0

    def getOutput(self, input):
        output = input[:,0]*input[:,0]*input[:,0]
        return (output,self.getFlags(output))

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"^3)"

    def getSymbolicOutputConstant(self, input):
        return ("("+str(input[0])+"^3)",0)

    def copy(self):
        return Cube()

    def propagateUnits(self, input):
        return 3*input[0]

class AddConstant(BaseWithConstants):
    numInputs = 1

    def getLatex(self):
        return "+c"

    def getOutput(self, input):
        return (input[:,0]+self.constant,0)

    def getSymbolicOutput(self, input):
        if type(self.constant) is not int:
            return "("+str(input[0])+"+"+str(round(self.constant.item(),3))+")"
        return "("+str(input[0])+"+"+str(round(self.constant,3))+")"

    def getSymbolicOutputConstant(self, input):
        if abs(self.constant) <= 0.1:
            return ("("+str(input[0])+"+"+self.symbolicConstant+")",1)
        return ("("+str(input[0])+"+"+self.symbolicConstant+")",0)

    def copy(self):
        return AddConstant()

    def propagateUnits(self, input):
        return input[0]


class MultiplyConstant(BaseWithConstants):
    numInputs = 1
    
    def getLatex(self):
        return "*c"

    def getOutput(self, input):
        return (input[:,0]*self.constant,0)

    def getSymbolicOutput(self, input):
        if type(self.constant) is not int:
            return "("+str(round(self.constant.item(),3))+"*"+str(input[0])+")"
        return "("+str(round(self.constant,3))+"*"+str(input[0])+")"

    def getSymbolicOutputConstant(self, input):
        if abs(self.constant) <= 0.1 or abs(self.constant-1) <= 0.1:
            return ("("+self.symbolicConstant+"*"+str(input[0])+")",1)
        return ("("+self.symbolicConstant+"*"+str(input[0])+")",0)
    
    def copy(self):
        return MultiplyConstant()
    
    def propagateUnits(self, input):
        return np.full(input[0].shape, np.nan)

class PowerConstant(BaseWithConstants):
    numInputs = 1
    
    def getLatex(self):
        return "x^c"

    def getOutput(self, input):
        output = torch.pow(input[:,0],self.constant)
        if torch.any(output!=output):
            return (output.detach(),self.getFlags(output))
        return (output,self.getFlags(output))

    def getFlags(self, output):
        if torch.all(torch.abs(output-1)<0.5):
            return min(5, 0.1/torch.mean(torch.abs(output-1)).detach())
        return 0

    def getSymbolicOutput(self, input):
        if type(self.constant) is not int:
            return "("+str(input[0])+"^"+str(round(self.constant.item(),3))+")"
        return "("+str(input[0])+"^"+str(round(self.constant,3))+")"

    def getSymbolicOutputConstant(self, input):
        if abs(self.constant) <= 0.1 or abs(self.constant-1) <= 0.1:
            return ("("+str(input[0])+"^"+self.symbolicConstant+")",1)
        return ("("+str(input[0])+"^"+self.symbolicConstant+")",0)

    def copy(self):
        return PowerConstant()

    def propagateUnits(self, input):
        return checkNonzero(input[0])

