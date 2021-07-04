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

