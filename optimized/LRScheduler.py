class ExpScheduler:
    def __init__(initialRate, decay):
        self.rate = initialRate
        self.decay = decay

    def getRate(epoch):
        return (self.rate*(self.decay**epoch))
