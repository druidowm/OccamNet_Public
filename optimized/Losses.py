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

class Adaptive:
    def __init__(self, stdScale, topNumber, anomWeight = 0.2):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.stdScale=stdScale
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)], device = self.device)
        self.anomWeight = anomWeight

    def getError(self, y, predictions):
        anom = (predictions != predictions)
        predictions[anom] = 0

        predictionDiff = predictions-y.unsqueeze(1)
        std = self.stdScale*torch.quantile(torch.abs(predictionDiff),0.5,dim=1,keepdim=True)+0.000001
        #print(std[:8,0,0])

        gaussian = torch.exp(-predictionDiff**2/(2*std))#*std

        error = torch.sum(gaussian, 0)

        anom = torch.any(anom, 0)

        error[anom] = -self.anomWeight*y.shape[0]

        return error

    def getLossMultipleSamples(self, logprobs, y, predictions):
        with torch.no_grad():
            error = self.getError(y, predictions)
        #print(error)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = torch.gather(error, 0, topLoc)
        logprobs = torch.gather(logprobs, 0, topLoc)

        return (torch.sum(torch.sum(-logprobs*error,axis=1)*self.weighting[0:logprobs.shape[0]]),torch.mean(error))


class Adaptive2:
    def __init__(self, stdScale, topNumber, anomWeight = 0.2):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.stdScale=stdScale
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)], device = self.device)
        self.anomWeight = anomWeight

    def getError(self, y, predictions):
        anom = (predictions != predictions)
        predictions[anom] = 0

        predictionDiff = predictions-y.unsqueeze(1)
        std = self.stdScale*torch.quantile(torch.abs(predictionDiff),0.5,dim=1,keepdim=True)+0.000001

        fold = 2-torch.abs(predictionDiff)/std#gaussian = torch.exp(-predictionDiff**2/(2*std))#*std

        error = torch.sum(fold, 0)

        anom = torch.any(anom, 0)

        error[anom] = -self.anomWeight*y.shape[0]

        return error

    def getLossMultipleSamples(self, logprobs, y, predictions):
        with torch.no_grad():
            error = self.getError(y, predictions)
        #print(error)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = torch.gather(error, 0, topLoc)
        logprobs = torch.gather(logprobs, 0, topLoc)

        return (torch.sum(torch.sum(-logprobs*error,axis=1)*self.weighting[0:logprobs.shape[0]]),torch.mean(error))


class Adaptive3:
    def __init__(self, stdScale, topNumber, anomWeight = 0.2):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.stdScale=stdScale
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)], device = self.device)
        self.anomWeight = anomWeight

    def getError(self, y, predictions):
        anom = (predictions != predictions)
        predictions[anom] = 0

        predictionDiff = predictions-y.unsqueeze(1)
        std = self.stdScale*torch.quantile(torch.abs(predictionDiff),0.5,dim=1,keepdim=True)+0.000001

        fold = 2-torch.abs(predictionDiff)/std#gaussian = torch.exp(-predictionDiff**2/(2*std))#*std

        error = torch.sum(fold, 0)

        anom = torch.any(anom, 0)

        error[anom] = -self.anomWeight*y.shape[0]

        return error

    def getLossMultipleSamples(self, logprobs, y, predictions):
        with torch.no_grad():
            error = self.getError(y, predictions)
        #print(error)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = torch.gather(error, 0, topLoc)
        logprobs = torch.gather(logprobs, 0, topLoc)

        return (torch.sum(torch.sum(-logprobs*error,axis=1)*self.weighting[0:logprobs.shape[0]]),torch.mean(error))


class Adaptive4:
    def __init__(self, stdScale, topNumber, anomWeight = 0.2):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.stdScale=stdScale
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)], device = self.device)
        self.anomWeight = anomWeight

    def getError(self, y, predictions):
        anom = (predictions != predictions)
        predictions[anom] = 0

        predictionDiff = predictions-y.unsqueeze(1)
        # = self.stdScale*torch.quantile(torch.abs(predictionDiff),0.5,dim=1,keepdim=True)+0.000001

        #fold = 2-torch.abs(predictionDiff)/std#gaussian = torch.exp(-predictionDiff**2/(2*std))#*std

        log = -torch.log(torch.abs(predictionDiff)+0.0001)

        error = torch.sum(log, 0)

        anom = torch.any(anom, 0)

        error[anom] = -self.anomWeight*y.shape[0]

        return error

    def getLossMultipleSamples(self, logprobs, y, predictions):
        with torch.no_grad():
            error = self.getError(y, predictions)
        #print(error)
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:]
        error = torch.gather(error, 0, topLoc)
        logprobs = torch.gather(logprobs, 0, topLoc)

        return (torch.sum(torch.sum(-logprobs*error,axis=1)*self.weighting[0:logprobs.shape[0]]),torch.mean(error))
