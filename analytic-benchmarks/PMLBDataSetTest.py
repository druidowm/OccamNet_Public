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


def geneticGrid(train_X, train_Y, name):
    for i in range(10):
        out = geneticRun([400,4,0.5], train_X, train_Y)

        f = open(f"eplex{name}.txt","a")
        f.write(out)
        f.close()

def feynmanGrid(train_X, train_Y, name):
    for i in range(10):
        out = AFR(train_X, train_Y, [100,1,500])

        f = open(f"aif{name}.txt","a")
        f.write(out)
        f.close()

x=random.rand(100,1)
feynmanGrid(x,x,"test")