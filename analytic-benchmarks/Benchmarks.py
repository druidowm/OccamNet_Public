from pmlb import fetch_data, regression_dataset_names
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
from numpy import random

from GeneticAlgorithm import geneticRun
from Feynman import aiFeynmanRun as AFR

import torch
import torch.nn as nn

import operator
import math
import random
import time
import argparse

def protectedDiv(left, right):
    with np.errstate(divide='ignore',invalid='ignore'):
        div = left / right
        div[np.isinf(div)] = 1
        div[np.isnan(div)] = 1
    return div

def protectedLog(item):
    with np.errstate(divide='ignore',invalid='ignore'):
        log = np.log(np.abs(item))
        log[np.isinf(log)] = 1
        log[np.isnan(log)] = 1
    return log


def genetic(train_X, train_Y, name, depth, basis):
    train_X = [train_X[:,i] for i in range(train_X.shape[1])]
    for i in range(10):
        out = geneticRun([400,depth,0.5], train_X, train_Y, basis)

        f = open(f"eplex{name}.txt","a")
        f.write(str(out)+"\n")
        f.close()

def feynman(train_X, train_Y, name):
    train_X = [train_X[:,i] for i in range(train_X.shape[1])]
    for i in range(10):
        out = AFR(train_X, train_Y, [100,0,500])

        f = open(f"aif{name}.txt","a")
        f.write(out+"\n")
        f.close()

x=np.random.rand(100,1)
#feynman(x,x[:,0],"test")
genetic(x,x[:,0],"test", 3, [(np.add, 2),(np.subtract, 2),(np.multiply, 2),(protectedDiv, 2),(np.cos, 1),(np.sin, 1),(np.exp, 1),(protectedLog, 1)])