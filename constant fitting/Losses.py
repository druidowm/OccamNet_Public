import torch
import math

class CrossEntropyLoss:
    def __init__(self, std, topNumber, anomWeight = 0.2):
        self.setStd(std)
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)])
        self.anomWeight = anomWeight

    def setStd(self, std):
        self.std = std
        self.scaleFactor = 1/math.sqrt(2*math.pi*(std**2))
    
    def getError(self, y, predictions):
        gaussian = torch.distributions.Normal(y, self.std)

        anom = (predictions != predictions)
        predictions[anom] = 0

        error = torch.sum(torch.exp(torch.sum(gaussian.log_prob(predictions), 1)), 0)

        anom = torch.any(torch.any(anom, 1), 0)

        error[anom] = -self.anomWeight*y.shape[0]*(self.scaleFactor**y.shape[1])

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
        return (torch.sum(-torch.log(probs)*error*self.weighting[0:probs.shape[0]]),torch.mean(error))
