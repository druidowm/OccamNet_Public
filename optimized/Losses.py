import torch
import math

class CEL:
    def __init__(self, std, topNumber, anomWeight = 0.2):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.setStd(std)
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)], device = self.device)
        self.anomWeight = anomWeight

    def setStd(self, std):
        self.std = std
        self.scaleFactor = 1/math.sqrt(2*math.pi*(std**2))

    def getError(self, y, predictions):
        gaussian = torch.distributions.Normal(y.unsqueeze(2), self.std)

        anom = (predictions != predictions)
        predictions[anom] = 0

        error = torch.sum(torch.exp(torch.sum(gaussian.log_prob(predictions), 2)), 0)

        anom = torch.any(torch.any(anom, 2), 0)

        error[anom] = -self.anomWeight*y.shape[0]*(self.scaleFactor**y.shape[1])

        return error

    def getLossMultipleSamples(self, logprobs, y, predictions):
        error = self.getError(y, predictions)
        #print(error)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = error[topLoc]
        logprobs = logprobs[topLoc]

        return (torch.sum(-logprobs*error*self.weighting[0:logprobs.shape[0]]),torch.mean(error))
