from sympy import *
import torch
import math

# BASIS FUNCTIONS DEFINITION
SIGMOID_PRECISION = 1000000
NORMAL_VARIANCE = 0.05

𝓝 = lambda x: torch.exp(-0.5 * (x)**2 / NORMAL_VARIANCE)
σ = lambda x: torch.sigmoid(SIGMOID_PRECISION * x)

def if_(x):
    if x == True: return 1
    if x == False: return 0
    return Function('if')(x)

def inequality_(x, y, z):
    if y == z: return y
    return if_(x > 0) * y  + if_(x <= 0) * z

def xor_(x, y):
    return Function('XOR')(x, y)

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
MAX            = ['MAX', (lambda x, y: torch.max(x, y)), (lambda x, y: Max(x, y)), '$\max$']
MIN            = ['MIN', (lambda x, y: torch.min(x, y)), (lambda x, y: Min(x, y)), '$\min$']
INEQUALITY2     = ['INEQUALITY2', (lambda o, x, y, z: σ(x - o) * y + (1 - σ(x- o)) * z),
                        (lambda o, x, y, z: inequality_(x - o, y, z)), "$\leq$"]
INEQUALITY     = ['INEQUALITY', (lambda x, y, z: σ(x) * y + (1 - σ(x)) * z),
                        (lambda x, y, z: inequality_(x, y, z)), "$\leq$"]
EQUALITY       = ['EQUALITY', (lambda x, y, z: 𝓝(x) * y + (1 - 𝓝(x)) * z),
                        (lambda x, y, z: if_(x == 0) * y  + if_(x != 0) * z), "$=$"]
XOR            = ['XOR', (lambda x, y: ((x).type(torch.BoolTensor) ^ (y).type(torch.BoolTensor)).type(torch.FloatTensor)), (lambda x, y: xor_(x, y)), '$⊻$']


COMPLETE_SET = [ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, MODULO, SQUARE, MAX, MIN, XOR,
                SINE, IDENTITY, NEGATIVE, INEQUALITY, EQUALITY, INEQUALITY2]

TORCH_BASES = { f[0]: f[1] for f in COMPLETE_SET }
SYMPY_BASES = { f[0]: f[2] for f in COMPLETE_SET }
LATEX_BASES = { f[0]: f[3] for f in COMPLETE_SET }

# ===========================================
### CONSTANTS COME HERE
# ===========================================


ONE            = ['ONE', 1, "$1$"]
TWO            = ['TWO', 2, "$2$"]
TEN            = ['TEN', 10, "$10$"]
PI             = ['PI',  math.pi, "$\pi$"]
E              = ['E',  math.e, "$e$"]

CONSTANTS_SET = [ONE, TWO, PI, E]
TORCH_CONSTANTS = { f[0]: f[1] for f in CONSTANTS_SET }
LATEX_CONSTANTS = { f[0]: f[2] for f in CONSTANTS_SET }
