from sympy import *
import torch.nn as nn
from inspect import signature
import torch.nn.functional as F
import torch
from bases import *


def get_model_equation(model, arg_max=True):
    def argmax_matrix(M):
        argmaxes = torch.argmax(M, dim=1).unsqueeze(-1)
        matrix = torch.zeros_like(M)
        for i, argmax in enumerate(argmaxes):
            matrix[i, argmax] = 1
        return matrix

    inputs = symbols(['x_' + str(i) for i in range(model.number_of_variables)])
    constants = symbols([LATEX_CONSTANTS[constant][1:][:-1] for constant in model.constants])
    outputs = symbols(['y_' + str(i) for i in range(model.number_of_outputs)])

    bases = [SYMPY_BASES[base] for base in model.bases]

    layers = []
    for module in model.children():
        layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]

    source = layers[0]
    source_w = source.weight.detach()

    number_of_inputs = source_w.shape[1]
    input_variables = Matrix(inputs + constants)

    source_w = F.softmax((1.0 / model.temperature) * source_w, dim=1)
    source_w = argmax_matrix(source_w)
    source_w = Matrix(source_w.detach().numpy().astype(int))

    args = source_w * input_variables
    past_imgs = inputs + constants

    for layer in layers[1:]:
        args_idx, img = 0, []
        for f in bases:
            arity = get_arity(f)
            arg = args[args_idx: args_idx + arity]
            img.append(f(*arg))
            args_idx = args_idx + arity

        # print(img)
        if model.skip_connections: img = Matrix(img + past_imgs)
        else: img = Matrix(img)
        # print(img)
        past_imgs = img[:]
        W = layer.weight.detach()
        W = F.softmax((1.0 / model.temperature) * W, dim=1)
        W = argmax_matrix(W)
        W = Matrix(W.detach().numpy().astype(int))
        # print("GOT HERE")
        args = W * img
        # print("ARGS", args)


    args = [simplify(arg) for arg in args]
    return args

def get_arity(f):
    return len(signature(f).parameters)
