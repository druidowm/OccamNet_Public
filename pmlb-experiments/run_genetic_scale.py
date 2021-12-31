from GeneticAlgorithm import geneticRun, geneticRunFullData
import time
import pickle
import sys
import argparse

def geneticGridFullData(parameters, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    train_X = [train_X[:,i] for i in range(train_X.shape[1])]
    val_X = [val_X[:,i] for i in range(val_X.shape[1])]
    test_X = [test_X[:,i] for i in range(test_X.shape[1])]

    params = []
    i=0
    for popSize in parameters[0]:
        for epochs in parameters[1]:
            for constraint in parameters[2]:
                for crossover in parameters[3]:
                    params.append([popSize,epochs,constraint,crossover,i])
                    i+=1
    outs = [geneticRunFullData(param, train_X, train_Y, val_X, val_Y, test_X, test_Y) for param in params]

    train = []
    val = []
    test = []
    trainFunction = []
    valFunction = []
    times = []
    for data in outs:
        e,runTime = data
        times.append(runTime)
        train.append(e.bestTrain)
        trainFunction.append(str(e.bestTrainFunction))
        val.append(e.bestVal)
        valFunction.append(str(e.bestValFunction))
        test.append(e.bestTest)

    return (parameters,train,val,test,trainFunction,valFunction,times)


def main(i):
    file = open('pmlb.dat', 'rb')
    data = pickle.load(file)
    file.close()
    item = data[i]
    print(item[0])
    trainData = item[1]
    f = open("geneticPMLB.txt", "a")
    f.write(f"started {i}\n")
    f.close()

    result = geneticGridFullData([[250, 500, 1000, 2000, 4000], [1000], [4], [0.2,0.5,0.8]],#,0.5,0.8]],
                          trainData[0], trainData[1], trainData[2], trainData[3], trainData[4], trainData[5])
    with open(f"geneticPMLB{i}.dat", "wb") as f:
        pickle.dump(result, f)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    main(int(args.dataset))