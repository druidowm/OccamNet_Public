from Losses import CrossEntropyLoss as L1
from Losses import CrossEntropyLoss2 as L2
import torch
import math

y = torch.tensor([[1,2],[2.,3]])

pred = torch.tensor([[[1.,2],[3,2]],[[2,1],[4,3]]])

probs1 = torch.tensor([1.,1])/math.e
probs2 = torch.tensor([[2.,1],[1,2]])/math.e

l1 = L1(1.0,2)
l2 = L2(1.0,2)
print(l1.getLossMultipleSamples(probs1,y,pred))
print(l2.getLossMultipleSamples(probs2,y,pred))