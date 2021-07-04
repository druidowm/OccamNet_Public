import AIFeynman.aifeynman as aif
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
import os,shutil
#import aifeynman as aif

def saveFile(X,y):
    with open("ai_feynman_dataset.txt", "w") as file:
        string = ""
        for i in range(0,X[0].shape[0]):
            for j in range(0,len(X)):
                string += str(X[j][i])+" "
            string += str(y[i])+"\n"
        file.write(string)

def getEqn(line, numVars):
    for _ in range(5):
        line = line[line.find(" ")+1:]
    eqn = line[:-1]

    eqn = eqn.replace("log","np.log")
    eqn = eqn.replace("exp","np.exp")
    eqn = eqn.replace("cos","np.cos")
    eqn = eqn.replace("sin","np.sin")
    eqn = eqn.replace("sqrt","np.sqrt")

    vars = "x0"
    for i in range(1,numVars):
        vars += ", x"+str(i)

    expr = "functions.append(((lambda "+vars+" : "+eqn+"+0*x0), \""+eqn+"\"))"
    print(expr)
    return compile(expr,"eqn","exec")

def readData(numVars):
    functions = []
    with open("results/solution_ai_feynman_dataset.txt","r") as f:
        for line in f.readlines():
            eqn = getEqn(line,numVars)
            exec(eqn)
    return functions

def aiFeynmanRun(train_X,train_Y,val_X,val_Y,params):
    saveFile(train_X,train_Y)
    aif.run_aifeynman("./", "ai_feynman_dataset.txt", params[0], "feynmanBasis.txt", polyfit_deg=params[1], NN_epochs=params[2],test_percentage = 0)
    #aif.run_aifeynman("./", "example_data/example2.txt", params[0], "feynmanBasis.txt", polyfit_deg=params[1], NN_epochs=params[2],test_percentage = 0)
    functions = readData(len(train_X))
    
    try:
        shutil.rmtree("results")
    except:
        pass
    try:
        os.remove("ai_feynman_dataset.txt_train")
    except:
        pass

    try:
        os.remove("ai_feynman_dataset.txt_test")
    except:
        pass

    try:
        os.remove("ai_feynman_dataset.txt")
    except:
        pass

    try:
        os.remove("args.dat")
    except:
        pass
    
    try:
        os.remove("mystery.dat")
    except:
        pass

    try:
        os.remove("results.dat")
    except:
        pass

    try:
        os.remove("results_gen_sym.dat")
    except:
        pass

    try:
        os.remove("qaz.dat")
    except:
        pass

    print(functions)

    bestTrainFunction = None
    bestTrainError = float("inf")

    bestValFunction = None
    bestValError = float("inf")

    for function in functions:
        try:
            trainError = MSE(train_Y, function[0](*train_X))
        except:
            trainError = float("inf")
        
        try:
            valError = MSE(val_Y, function[0](*val_X))
        except:
            valError = float("inf")

        if (trainError < bestTrainError):
            bestTrainError = trainError
            bestTrainFunction = function

        if (valError < bestValError):
            bestValError = valError
            bestValFunction = function

    return (bestTrainFunction, bestTrainError, bestValFunction, bestValError)