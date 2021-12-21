from pmlb import fetch_data, regression_dataset_names
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np
from numpy import random

from GeneticAlgorithm import geneticRun
#from Feynman import aiFeynmanRun as AFR
from dso import DeepSymbolicRegressor as DSO

import torch
import torch.nn as nn

import operator
import math
import random
import time
import argparse

import datetime

import json

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


def parse(function, numIn):
    function = function.replace("ARG","x")
    function = function.replace("add","np.add")
    function = function.replace("subtract","np.subtract")
    function = function.replace("multiply","np.multiply")
    function = function.replace("divide","np.divide")
    function = function.replace("negative","np.negative")
    function = function.replace("sin","np.sin")
    expression = "funcs.append(lambda "

    for i in range(numIn):
        expression += f"x{i}, "
    
    expression = expression[:-2] + ": " + function+")"
    print(expression)
    funcs = []
    exec(expression)
    return funcs[0]

def genetic(train_X, train_Y, name, depth, basis, popsize, epochs):
    times = []
    train_X = [train_X[:,i] for i in range(train_X.shape[1])]
    numCorrect = 0
    print(basis)

    plt.scatter(train_X[0], train_Y)
    plt.scatter(train_X[0], inequality(train_X[0],3*train_X[0],train_X[0]))
    plt.show()

    for i in range(10):
        now1 = datetime.datetime.now()
        out = geneticRun([popsize,depth,0.5], train_X, train_Y, basis, epochs)
        now2 = datetime.datetime.now()
        times.append(now2-now1)

        f = open(f"eplex{name}.txt","a")
        f.write(str(out)+"\n")
        f.close()

        func = parse(str(out),len(train_X))
        if np.all(np.isclose(func(*train_X),train_Y)):
            numCorrect+=1
        
    f = open(f"eplex{name}.txt","a")
    f.write(str(numCorrect)+"\n")
    f.write(str(np.mean(times)))
    f.close()

def feynman(train_X, train_Y, name, basis):
    with open("AIFeynman/aifeynman/feynmanBasis.txt", "w") as f:
        f.write(basis)

    times = []
    train_X = [train_X[:,i] for i in range(train_X.shape[1])]
    for i in range(10):
        now1 = datetime.datetime.now()
        out = AFR(train_X, train_Y, [100,0,500])
        now2 = datetime.datetime.now()
        times.append(now2-now1)

        f = open(f"aif{name}.txt","a")
        f.write(out+"\n")
        f.close()
    f = open(f"aif{name}.txt","a")
    f.write(str(np.mean(times)))
    f.close()

def deepSymbolic(train_X, train_Y, name, depth, basis, popsize, epochs):
    data = {"task" : {"task_type" : "regression", "function_set" : basis},
            "training" : {"batch_size" : popsize}}

    with open('config.json', 'w') as outfile:
        json.dump(data, outfile)

    times = []
    numCorrect = 0

    plt.scatter(train_X[:,0], train_Y)
    plt.show()

    for i in range(10):
        now1 = datetime.datetime.now()
        model = DSO("config.json")
        model.fit(train_X, train_Y)
        now2 = datetime.datetime.now()
        times.append(now2-now1)

        f = open(f"dsr{name}.txt","a")
        f.write(str(model.program_.pretty())+"\n")
        f.close()

        if np.all(np.isclose(model.predict(train_X),train_Y)):
            numCorrect+=1
        
    f = open(f"dsr{name}.txt","a")
    f.write(str(numCorrect)+"\n")
    f.write(str(np.mean(times)))
    f.close()

def batchRunGenetic(equations, inputs, constants, floors, popsizes, bases, maxDepths):
    i=1
    for equation, input, constant, floor, popsize, base, maxDepth in zip(equations, inputs, constants, floors, popsizes, bases, maxDepths):
        size = 200
        xs = []

        for item in input:
            top = item[0]
            bottom = item[1]
            if floor:
                xs.append(np.floor((top-bottom)*np.random.rand(size,1)+bottom).astype(int))
            else:
                xs.append((top-bottom)*np.random.rand(size,1)+bottom)

        y = equation(*xs)
        xs = xs+[np.full((size,1),const) for const in constant]
        valid = np.isfinite(y)[:,0]
        y = y[valid,:]
        xs = [item[valid,:] for item in xs]
        y = y[:,0]
        x = np.concatenate(xs, 1)
        print(x)
        print(y)
        #feynman(x,y,"2","+*S")
        genetic(x,y,i, maxDepth, base, popsize, 1000)
        i+=1

def batchRunFeynman(equations, inputs, bases):
    i=1
    for equation, input, base in zip(equations, inputs, bases):
        size = 200
        xs = []

        for item in input:
            top = item[0]
            bottom = item[1]
            xs.append((top-bottom)*np.random.rand(size,1)+bottom)

        y = equation(*xs)[:,0]
        x = np.concatenate(xs, 1)
        feynman(x,y,i,base)
        i+=1

def batchRunDeepSymbolic(equations, inputs, constants, floors, popsizes, bases, maxDepths):
    i=1
    for equation, input, constant, floor, popsize, base, maxDepth in zip(equations, inputs, constants, floors, popsizes, bases, maxDepths):
        size = 200
        xs = []

        for item in input:
            top = item[0]
            bottom = item[1]
            if floor:
                xs.append(np.floor((top-bottom)*np.random.rand(size,1)+bottom).astype(int))
            else:
                xs.append((top-bottom)*np.random.rand(size,1)+bottom)

        y = equation(*xs)
        xs = xs
        valid = np.isfinite(y)[:,0]
        y = y[valid,:]
        xs = [item[valid,:] for item in xs]
        y = y[:,0]
        x = np.concatenate(xs, 1)
        print(x)
        print(y)
        base += constant
        deepSymbolic(x,y,i, maxDepth, base, popsize, 1000)
        i+=1

def f1(x):
    y = 1*x
    y[x>0] = 3*x[x>0]
    return y

def f2(x):
    y=-x
    y[x>0] = x[x>0]**2
    return y

def f3(x):
    y = 1*x
    y[x<=0] = np.sin(x[x<=0])
    return y

def f41(x,y,z):
    return np.maximum(np.maximum(x,y),z)

def f42(x,y,z):
    out = []
    for i in range(x.shape[0]):
        out.append(np.median([x[i],y[i],z[i]]))
    return np.expand_dims(np.array(out),1)

def f43(x,y,z):
    return np.minimum(np.minimum(x,y),z)

def f51(w,x,y,z):
    return (w+z)%2

def f52(w,x,y,z):
    return w

def f53(w,x,y,z):
    return x

def f54(w,x,y,z):
    return y

def f61(x,y):
    z = 1*y
    z[x>=2] = -y[x>=2]
    return z

def f62(x,y):
    z = 1*x
    z[x>=0] = (y**2)[x>=0]
    return z

def g1(x):
    y = x/2
    y[x<2] = (x**2)[x<2]
    return y

def f7(x):
    return g1(g1(g1(g1(x))))

def g2(x):
    y = x-1
    y[x<2] = (x+2)[x<2]
    return y

def f8(x):
    return g2(g2(x))

def inequality(x, y, z):
    out = np.copy(y)
    out[x<=0] = z[x<=0]
    return out

def MIN(x,y):
    return np.minimum(x,y)

def MAX(x,y):
    return np.maximum(x,y)

def XOR(x,y):
    return np.logical_xor(x,y)


equations = [lambda x: 2*x**2+3*x, 
             lambda x: np.sin(3*x+2),
             lambda x: np.sin(x)+np.sin(2*x)+np.sin(3*x),
             lambda x: (x**2+x)/(x+2),
             lambda x,y: x**2*(x+1)/(y**5),
             lambda x,y: x**2/2+(y+1)**2/2,
             f1,
             f2,
             f3,
             f41,
             f42,
             f43,
             f51,
             f52,
             f53,
             f54,
             f61,
             f62,
             f7,
             f8]

inputs = [[(-10,10)],
          [(-10,10)],
          [(-20,20)],
          [(-6,6)],
          [(-10,10),(0.1,3)],
          [(-20,-2),(2,20)],
          [(-20,20)],
          [(-20,20)],
          [(-20,20)],
          [(-50,50),(-50,50),(-50,50)],
          [(-50,50),(-50,50),(-50,50)],
          [(-50,50),(-50,50),(-50,50)],
          [(0,2),(0,2),(0,2),(0,2)],
          [(0,2),(0,2),(0,2),(0,2)],
          [(0,2),(0,2),(0,2),(0,2)],
          [(0,2),(0,2),(0,2),(0,2)],
          [(-5,5),(-5,5)],
          [(-5,5),(-5,5)],
          [(-8,8)],
          [(-3,6)]]

floor = [False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,False,False,False,False]

constants = [[],
             [1,2],
             [1,2],
             [1],
             [1],
             [1,2],
             [1],
             [1],
             [1],
             [1,2],
             [1,2],
             [1,2],
             [],
             [],
             [],
             [],
             [1,2],
             [1,2],
             [1,2],
             [1,2]]

popsizes = [50, 50, 50, 100, 100, 150, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

basesEplex = [[(np.add, 2),(np.multiply, 2)],
         [(np.add, 2),(np.multiply, 2),(np.sin, 1)],
         [(np.add, 2),(np.sin, 1)],
         [(np.add, 2),(np.multiply, 2),(protectedDiv, 2)],
         [(np.add, 2),(np.multiply, 2),(protectedDiv,2)],
         [(np.add, 2),(np.multiply, 2),(protectedDiv,2)],
         [(np.add, 2),(np.multiply, 2),(protectedDiv, 2),(inequality,3)],
         [(np.add, 2),(np.subtract,2),(np.negative,1),(np.multiply, 2),(inequality,3)],
         [(np.add, 2),(np.sin, 2),(inequality,3)],
         [(np.add, 2),(np.subtract, 2),(np.multiply,2),(inequality,3),(MIN,2),(MAX,2)],
         [(np.add, 2),(np.subtract, 2),(np.multiply,2),(inequality,3),(MIN,2),(MAX,2)],
         [(np.add, 2),(np.subtract, 2),(np.multiply,2),(inequality,3),(MIN,2),(MAX,2)],
         [(np.add, 2),(XOR,2)],
         [(np.add, 2),(XOR,2)],
         [(np.add, 2),(XOR,2)],
         [(np.add, 2),(XOR,2)],
         [(np.multiply, 2),(np.negative, 1),(inequality,3)],
         [(np.multiply, 2),(np.negative, 1),(inequality,3)],
         [(np.add, 2),(np.multiply, 2),(protectedDiv,2),(inequality,3)],
         [(np.add, 2),(np.subtract, 2),(inequality,3)]]

basesDSR = [["add","mul"],
            ["add","mul","sin"],
            ["add","sin"],
            ["add","mul","div"],
            ["add","mul","div"],
            ["add","mul","div"]]

basesFeynman = ["+*","+*S1","+S1","+*D1","+*D1","+*D1"]

maxDepths = [3,3,5,3,4,4,3,3,3,6,6,6,10,10,10,10,4,4,12,10]

batchRunDeepSymbolic(equations[:6], inputs[:6], constants[:6], floor[:6], popsizes[:6], basesDSR[:6], maxDepths[:6])
#batchRunFeynman(equations[4:5], inputs[4:5], basesFeynman[4:5])