from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from inspect import signature

#from scoop import futures

import operator

import math
import random

import numpy as np

import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import multiprocessing
from functools import partial

from sklearn.metrics import mean_absolute_error, mean_squared_error

#import ray

def constantInitializer():
    return random.uniform(-10,10)

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

#@ray.remote
def geneticRun(params, train_X, train_Y, bases, epochs):
    print(params)
    popSize = params[0]
    constraint = gp.staticLimit(key=operator.attrgetter("height"), max_value=params[1])
    e = Eplex(len(train_X), bases, tools.selAutomaticEpsilonLexicase, constraint)
    e.runGeneticRegression(train_X, train_Y, popSize, epochs, params[2], 1-params[2])
    return e.bestTrainFunction

class Eplex:
    def evalSymbReg(self, individual):
        func = self.toolbox.compile(expr=individual)
        out = (func(*self.X) - self.y)**2
        out[out!=out]=100
        return np.sum(out)/self.y.shape[0],

    def __init__(self, numImputs, primitives, selectionType, constraint = None):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        pset = gp.PrimitiveSet("MAIN", numImputs)

        for item in primitives:
            pset.addPrimitive(item[0],item[1])

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

        self.toolbox.register("select", selectionType)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)

        if constraint != None:
            self.toolbox.decorate("mate", constraint)
            self.toolbox.decorate("mutate", constraint)


    def runGeneticRegression(self, X, y, popSize, epochs, cxpb, mutpb):
        self.X = X
        self.y = y

        self.toolbox.register("evaluate", self.evalSymbReg)

        pop = self.toolbox.population(n=popSize)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit)#, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb, mutpb, epochs, stats=mstats,
                                                   halloffame=hof, verbose=True)

        self.bestTrainFunction = None
        self.bestTrain = float("inf")

        self.bestValFunction = None
        self.bestVal = float("inf")
        for function in hof:
            train = self.getError(function, X, y)
            if train<self.bestTrain:
                self.bestTrain = train
                self.bestTrainFunction = function

    def getError(self, individual, X, y):
        func = self.toolbox.compile(expr=individual)
        out = func(*X)
        return mean_squared_error(y,out)

    def plotGraph(self, item):
        nodes, edges, labels = gp.graph(item)

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()
