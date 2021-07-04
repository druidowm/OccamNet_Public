import sympy as sp
from sympy import *

def checkEqualPermutations(args1,args2,opType,permute):


    argsNew1 = []
    argsNew2 = []

    for item in args1:
        expanded = expand(item)
        if expanded[0] == opType:
            for arg in expanded[1]:
                argsNew1.append(arg)
        else:
            argsNew1.append(item)

    for item in args2:
        expanded = expand(item)
        if expanded[0] == opType:
            for arg in expanded[1]:
                argsNew2.append(arg)
        else:
            argsNew2.append(item)

    if len(argsNew1) != len(argsNew2):
        return False
    
    if len(argsNew1) == 1:
        return sympyEquals(argsNew1[0],argsNew2[0],permute)

    for i in range(len(argsNew1)):
        if sympyEquals(argsNew1[0],argsNew2[i],permute):
            if checkEqualPermutations(argsNew1[1:],argsNew2[:i]+argsNew2[i+1:],opType,permute):
                return True

    return False

def equals(args1,args2,op,permute):
    if op == "Add" or op == "Mul":
        argsNew1 = []
        argsNew2 = []

        for item in args1:
            expanded = expand(item)
            if expanded[0] == op:
                for arg in expanded[1]:
                    argsNew1.append(arg)
            else:
                argsNew1.append(item)

        for item in args2:
            expanded = expand(item)
            if expanded[0] == op:
                for arg in expanded[1]:
                    argsNew2.append(arg)
            else:
                argsNew2.append(item)

        if len(argsNew1)==len(argsNew2):
            for i in range(len(argsNew1)):
                if not sympyEquals(argsNew1[i],argsNew2[i],permute):
                    return False
        else:
            return False

    else:
        if len(args1)==len(args2):
            for i in range(len(args1)):
                if not sympyEquals(args1[i],args2[i],permute):
                    return False
        else:
            return False
    return True


def expand(exp):
    if type(exp)==Mul:
        for i in range(len(exp.args)):
            if exp.args[i] == 2:
                newArg = Mul(*(exp.args[:i]+exp.args[i+1:]))
                return ("Add",[newArg,newArg])
            if exp.args[i] == 3:
                newArg = Mul(*(exp.args[:i]+exp.args[i+1:]))
                return ("Add",[newArg,newArg,newArg])

        return ("Mul",exp.args)

    if type(exp)==Pow:
        if exp.args[1].is_integer:
            if exp.args[1]>0:
                return ("Mul",[exp.args[0] for i in range(exp.args[1])])

            else:
                if type(exp.args[0])==Mul:

                    for j in range(len(exp.args[0].args)):
                        if exp.args[0].args[j] == 2:

                            powArgs = exp.args[0].args[:j]+exp.args[0].args[j+1:]

                            return ("Pow",[Add(UnevaluatedExpr(powArgs),UnevaluatedExpr(powArgs)),exp.args[1]])

                    newArgs = []
                    for item in exp.args[0].args:
                        newArgs.append(pow(item,exp.args[1]))
                    return ("Mul",newArgs)

    if type(exp) == Add:
        return ("Add",exp.args)

    if type(exp) == Pow:
        return ("Pow",exp.args)

    if type(exp) == cos:
        return ("Cos",exp.args)

    if type(exp) == sin:
        return ("Sin",exp.args)

    return ("Other",exp.args)



def sympyEquals(exp1, exp2, permute):
    if exp1 == exp2:
        return True

    type1,args1 = expand(exp1)
    type2,args2 = expand(exp2)

    if len(args1)!=0 and len(args2)!=0:
        if type1 == type2:
            if permute and (type1=="Add" or type1=="Mul") and (type2=="Add" or type2=="Mul"):
                if checkEqualPermutations(args1,args2,type1,permute):
                    return True
            else:
                if equals(args1,args2,type1,permute):
                    return True

            

    return False


