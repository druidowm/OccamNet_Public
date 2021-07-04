import torch
import math

class CrossEntropyLoss:
    def __init__(self, var, topNumber):
        self.setVar(var)
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)])

    def setVar(self, var):
        self.var = var
    
    def getError(self, y, predictions):
        gaussian = torch.distributions.Normal(y, self.var)
        error = torch.sum(torch.exp(torch.sum(gaussian.log_prob(predictions), 1)), 0)
        return error

    def getErrorRecursion(self, y, predictions):
        return self.getError(y.unsqueeze(2).repeat((1,1,predictions.shape[2])), predictions)

    def getLoss(self, prob, y, predictions):
        error = self.getErrorRecursion(y, predictions)
        return -torch.log(prob)*error

    def getLossMultipleSamples(self, probs, y, predictions):
        error = self.getErrorRecursion(y, predictions)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = error[topLoc]
        probs = probs[topLoc]
        return torch.sum(-torch.log(probs)*error*self.weighting)

class CELTrivialRegularization:
    def __init__(self, var, topNumber, trivialWeight):
        self.setVar(var)
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)])
        self.trivialWeight = trivialWeight

    def setVar(self, var):
        self.var = var
    
    def getError(self, y, predictions, trivial):
        gaussian = torch.distributions.Normal(y, self.var)
        error = torch.sum(torch.exp(torch.sum(gaussian.log_prob(predictions), 1)), 0)-self.trivialWeight*trivial
        return error

    def getErrorRecursion(self, y, predictions, trivial):
        return self.getError(y.unsqueeze(2).repeat((1,1,predictions.shape[2])), predictions, trivial)

    def getLoss(self, prob, y, predictions, trivial):
        error = self.getErrorRecursion(y, predictions, trivial)
        return -torch.log(prob)*error

    def getLossMultipleSamples(self, probs, y, predictions, trivial):
        error = self.getErrorRecursion(y, predictions, trivial)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = error[topLoc]
        probs = probs[topLoc]
        return torch.sum(-torch.log(probs)*error*self.weighting)

class CELTrivialConstantRegularization:
    def __init__(self, var, topNumber, trivialWeight, constantWeight):
        self.setVar(var)
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)])
        self.trivialWeight = trivialWeight
        self.constantWeight = constantWeight

    def setVar(self, var):
        self.var = var
    
    def getError(self, y, predictions, trivial, constant):
        gaussian = torch.distributions.Normal(y, self.var)
        error = torch.sum(torch.exp(torch.sum(gaussian.log_prob(predictions), 1)), 0)-self.trivialWeight*trivial-self.constantWeight*constant
        return error

    def getErrorRecursion(self, y, predictions, trivial, constant):
        return self.getError(y.unsqueeze(2).repeat((1,1,predictions.shape[2])), predictions, trivial, constant)

    def getLoss(self, prob, y, predictions, trivial, constant):
        error = self.getErrorRecursion(y, predictions, trivial, constant)
        return -torch.log(prob)*error

    def getLossMultipleSamples(self, probs, y, predictions, trivial, constant):
        error = self.getErrorRecursion(y, predictions, trivial, constant)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = error[topLoc]
        probs = probs[topLoc]
        return torch.sum(-torch.log(probs)*error*self.weighting)


class CELFlagRegularization:
    def __init__(self, var, topNumber, trivialWeight, flagWeight, constantWeight):
        self.setVar(var)
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)])
        self.trivialWeight = trivialWeight
        self.flagWeight = flagWeight
        self.constantWeight = constantWeight

    def setVar(self, var):
        self.var = var
    
    def getError(self, y, predictions, trivial, flags, constant):
        gaussian = torch.distributions.Normal(y, self.var)
        error = torch.sum(torch.exp(torch.sum(gaussian.log_prob(predictions), 1)), 0)
        error -= self.trivialWeight*trivial+self.flagWeight*flags+self.constantWeight*constant
        return error

    def getErrorRecursion(self, y, predictions, trivial, flags, constant):
        return self.getError(y.unsqueeze(2).repeat((1,1,predictions.shape[2])), predictions, trivial, flags, constant)

    def getLoss(self, prob, y, predictions, trivial, flags, constant):
        error = self.getErrorRecursion(y, predictions, trivial, flags, constant)
        return -torch.log(prob)*error

    def getLossMultipleSamples(self, probs, y, predictions, trivial, flags, constant):
        error = self.getErrorRecursion(y, predictions, trivial, flags, constant)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = error[topLoc]
        probs = probs[topLoc]
        return torch.sum(-torch.log(probs)*error*self.weighting)

class CELActivationRegularization:
    def __init__(self, std, topNumber, trivialWeight, flagWeight, constantWeight, activationWeight, verbosity = 1):
        self.setStd(std)
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)])
        self.trivialWeight = trivialWeight
        self.flagWeight = flagWeight
        self.constantWeight = constantWeight
        self.activationWeight = activationWeight
        self.verbosity = verbosity

    def setStd(self, std):
        self.std = std
        self.scaleFactor = 1/math.sqrt(2*math.pi*(std**2))
    
    def getError(self, y, predictions, trivial, flags, constant, activation):
        gaussian = torch.distributions.Normal(y, self.std)

        anom = (predictions != predictions)
        predictions[anom] = 0

        error = torch.sum(torch.exp(torch.sum(gaussian.log_prob(predictions), 1)), 0)

        anom = torch.any(torch.any(anom, 1), 0)

        error[anom] = -0.2*y.shape[0]*(self.scaleFactor**y.shape[1])


        error -= y.shape[0]*(self.scaleFactor**y.shape[1])*(self.trivialWeight*trivial+self.constantWeight*constant+self.activationWeight*activation)
        error2 = error - y.shape[0]*(self.scaleFactor**y.shape[1])*(self.flagWeight*flags)


        return (error2,error)

    def getErrorRecursion(self, y, predictions, trivial, flags, constant, activation):
        return self.getError(y.unsqueeze(2).repeat((1,1,predictions.shape[2])), predictions, trivial, flags, constant, activation)

    def getLoss(self, prob, y, predictions, trivial, flags, constant, activation):
        error = self.getErrorRecursion(y, predictions, trivial, flags, constant, activation)
        return -torch.log(prob)*error

    def getLossMultipleSamples(self, probs, y, predictions, trivial, flags, constant, activation):
        error,errorNoReg = self.getErrorRecursion(y, predictions, trivial, flags, constant, activation)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = error[topLoc]
        errorNoReg = errorNoReg[topLoc]
        probs = probs[topLoc]
        return (torch.sum(-torch.log(probs)*error*self.weighting[0:probs.shape[0]]),torch.mean(errorNoReg))
