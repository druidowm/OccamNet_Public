from sympy import *
import torch
import math

# BASIS FUNCTIONS DEFINITION
SIGMOID_PRECISION = 10
TANH_PRECISION = 10

# SIGMOID_PRECISION = 1
# TANH_PRECISION = 1

NORMAL_VARIANCE = 0.05

𝓝 = lambda x: torch.exp(-0.5 * (x)**2 / NORMAL_VARIANCE)
σ = lambda x: torch.sigmoid(SIGMOID_PRECISION * x)
tanh = lambda x: torch.tanh(TANH_PRECISION * x)
sigma = σ

def if_(x):
    if x == True: return 1
    if x == False: return 0
    return Function('if')(x)


def inequality_(x, y, z):
    if y == z: return y
    return if_(x > 0) * y  + if_(x <= 0) * z


def max_(x, y, ):
    return Function('max')(x, y)


def min_(x, y, ):
    return Function('min')(x, y)


def sigma_(x):
    return Function('sigma')(x)


def tanh_(x):
    return Function('tanh')(x)


def max9_(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return Function('max9')(x0, x1, x2, x3, x4, x5, x6, x7, x8)


def min9_(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return Function('min9')(x0, x1, x2, x3, x4, x5, x6, x7, x8)


def add9_(x0, x1, x2, x3, x4, x5, x6, x7, x8):
    return Function('add9')(x0, x1, x2, x3, x4, x5, x6, x7, x8)


def max4_(x0, x1, x2, x3):
    return Function('max4')(x0, x1, x2, x3)


def min4_(x0, x1, x2, x3):
    return Function('min4')(x0, x1, x2, x3)


def add4_(x0, x1, x2, x3):
    return Function('add4')(x0, x1, x2, x3)


# ===========================================
### BASES COME HERE
# ===========================================
ADDITION       = ['ADDITION', (lambda x, y: x + y), (lambda x, y: x + y), '$+$']
SUBTRACTION    = ['SUBTRACTION', (lambda x, y: x - y), (lambda x, y: x - y), '$-$']
MULTIPLICATION = ['MULTIPLICATION', (lambda x, y: x * y), (lambda x, y: x * y), '$\\times$']
DIVISION       = ['DIVISION', (lambda x, y: x / y), (lambda x, y: x / y), '$\div$']
MODULO         = ['MODULO', (lambda x, y: torch.remainder(x, y)), (lambda x, y: x % y), '$\%$']
SQUARE         = ['SQUARE',(lambda x: x**2), (lambda x: x**2), '$x^2$']
SINE           = ['SINE', (lambda x: torch.sin(x)), (lambda x: sin(x)), '$\sin$']
IDENTITY       = ['IDENTITY', (lambda x: x), (lambda x: x), '$\mathbf{I}$']
NEGATIVE       = ['NEGATIVE', (lambda x: -x), (lambda x: -x), '$-x$']
MAX            = ['MAX', (lambda x, y: torch.max(x, y)), (lambda x, y: max_(x, y)), '$\max$']

MAX4            = ['MAX4', (lambda x0, x1, x2, x3, x4:
                            torch.max(torch.cat([x0, x1, x2, x3]), dim=0)[0]
                            ),
                   (lambda x0, x1, x2, x3: max4_(x0, x1, x2, x3)), 'max9']

MIN4            = ['MIN4', (lambda x0, x1, x2, x3:
                            torch.min(torch.cat([x0, x1, x2, x3]), dim=0)[0]
                            ),
                   (lambda x0, x1, x2, x3: min4_(x0, x1, x2, x3)), 'min9']
ADD4            = ['ADD4', (lambda x0, x1, x2, x3:
                                x0 + x1 + x2 + x3
                            ),
                   (lambda x0, x1, x2, x3: add4_(x0, x1, x2, x3)), 'add9']


MAX9            = ['MAX9', (lambda x0, x1, x2, x3, x4, x5, x6, x7, x8:
                            torch.max(torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8]), dim=0)[0]
                            ),
                   (lambda x0, x1, x2, x3, x4, x5, x6, x7, x8: max9_(x0, x1, x2, x3, x4, x5, x6, x7, x8)), 'max9']


MIN9            = ['MIN9', (lambda x0, x1, x2, x3, x4, x5, x6, x7, x8:
                            torch.min(torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8]), dim=0)[0]
                            ),
                   (lambda x0, x1, x2, x3, x4, x5, x6, x7, x8: min9_(x0, x1, x2, x3, x4, x5, x6, x7, x8)), 'min9']
ADD9            = ['ADD9', (lambda x0, x1, x2, x3, x4, x5, x6, x7, x8:
                                x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
                            ),
                   (lambda x0, x1, x2, x3, x4, x5, x6, x7, x8: add9_(x0, x1, x2, x3, x4, x5, x6, x7, x8)), 'add9']

MIN            = ['MIN', (lambda x, y: torch.min(x, y)), (lambda x, y: min_(x, y)), '$\min$']
INEQUALITY2     = ['INEQUALITY2', (lambda o, x, y, z: σ(x - o) * y + (1 - σ(x- o)) * z),
                        (lambda o, x, y, z: inequality_(x - o, y, z)), "$\leq$"]
INEQUALITY     = ['INEQUALITY', (lambda x, y, z: σ(x) * y + (1 - σ(x)) * z),
                        (lambda x, y, z: inequality_(x, y, z)), "$\leq$"]
EQUALITY       = ['EQUALITY', (lambda x, y, z: 𝓝(x) * y + (1 - 𝓝(x)) * z),
                        (lambda x, y, z: if_(x == 0) * y  + if_(x != 0) * z), "$=$"]
XOR            = ['XOR', (lambda x, y: ((x).type(torch.BoolTensor) ^ (y).type(torch.BoolTensor)).type(torch.FloatTensor)), (lambda x, y: x ^ y), '$⊻$']
SIGMOID        = ['SIGMOID', lambda x: sigma(x), lambda x: sigma_(x), "$\sigma$"]
TANH           = ['TANH', lambda x: tanh(x), lambda x: tanh_(x), "tanh"]

COMPLETE_SET = [ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, MODULO, SQUARE, MAX, MIN, XOR,
                SINE, IDENTITY, NEGATIVE, INEQUALITY, EQUALITY, INEQUALITY2, SIGMOID, MAX9, MIN9, ADD9,  MAX4, MIN4, ADD4, TANH]

TORCH_BASES = { f[0]: f[1] for f in COMPLETE_SET }
SYMPY_BASES = { f[0]: f[2] for f in COMPLETE_SET }
LATEX_BASES = { f[0]: f[3] for f in COMPLETE_SET }

# ===========================================
### CONSTANTS COME HERE
# ===========================================


ZERO           = ['ZERO', 0, "$0$"]
ONE            = ['ONE', 1, "$1$"]
NONE            = ['NONE', -1, "$-1$"]
TWO            = ['TWO', 2, "$2$"]
NTWO            = ['NTWO', -2, "$-2$"]
TEN            = ['TEN', 10, "$10$"]
PI             = ['PI',  math.pi, "$\pi$"]
E              = ['E',  math.e, "$e$"]


CONSTANTS_SET = [ZERO, ONE, NONE, TWO, PI, E, NTWO]
TORCH_CONSTANTS = { f[0]: f[1] for f in CONSTANTS_SET }
LATEX_CONSTANTS = { f[0]: f[2] for f in CONSTANTS_SET }
