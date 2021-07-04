import torch
import numpy as np
import math



# ===========================================
### STANDARD
# ===========================================

IDENTITY = ['IDENTITY', lambda x: x]
CONSTANT_BUILDING = ['CONSTANT_BUILDING', lambda x: (3 * math.pi / (2 * math.e)) * x]
CONSTANT_MULTIPLY = ['CONSTANT_MULTIPLY', lambda x: 3 * x]
CONSTANT_INCREASE = ['CONSTANT_INCREASE', lambda x: 4 + x]
PROTOTYPING = ['PROTOTYPING', lambda x: (x) * (x > 2) + (-x) * (x <= 2)]
SQUARE = ['SQUARE', lambda x: x ** 2 + 2 * x]
SINE = ['SINE', lambda x: np.sin(x)]
SINE_SHIFT_OFFSET = ['SINE_SHIFT_OFFSET', lambda x: np.sin(2 * x + 1)]
BASIC_INEQUALITY = ['BASIC_INEQUALITY', lambda x: (x <= 0) * np.sin(2 * x) + np.sin(3 * x) * (x > 0)]
SIGNAL = ['SIGNAL', lambda x: np.sin(3 * x) + np.sin(2 * x) + np.sin(x)]
MULTI_INEQUALITY = ['MULTI_INEQUALITY',
                    lambda x: ((x[:, 1] <= 0) * x[:, 0] + x[:, 0] ** 2 * (x[:, 1] > 0)).unsqueeze(-1)]
BASIC_PRODUCT = ['BASIC_PRODUCT', lambda x: torch.prod(x, dim=-1).unsqueeze(-1)]
BASIC_SUM = ['BASIC_SUM', lambda x: torch.sum(x, dim=-1).unsqueeze(-1)]
INVERSE = ['INVERSE', lambda x: 1 / x]
SINE_DIFFERENT_FREQ = ['SINE_DIFFERENT_FREQ', lambda x: np.sin(3 * x)]
WEIRD_COUNTER = ['WEIRD_COUNTER', (lambda x: (x > 0) * 3 * x + (x <= 0) * 5 * x)]
WEIRD_PARABOLA = ['WEIRD_PARABOLA', (lambda x: (x > 0) * 2 * x + (x <= 0) * x ** 3)]
SINE_DIFFERENT_FREQ_PHASE_SQUARE = ['SINE_DIFFERENT_FREQ_PHASE_SQUARE', lambda x: np.sin(x) + x ** 2]
SQUARE_PLUS_CONSTANT_MULTIPLY = ['SQUARE_PLUS_CONSTANT_MULTIPLY', lambda x: x ** 2 + 2 * x]
MAX_FUNCTION = ['MAX_FUNCTION', lambda x: torch.max(x, dim=-1)[0].unsqueeze(-1)]
SIMPLE_RECURSION = ['SIMPLE_RECURSION', lambda x: x + 4]


# ===========================================
### PHYSICS
# ===========================================
GRAVITATION = ['GRAVITATION', lambda m1, m2, r: 6.6 * (m1 * m2 / r ** 2)]


# ===========================================
### COMPUTER SCIENCE
# ===========================================
SORTING = ['SORTING', lambda x: torch.sort(x)[0]]
NALU = ['NALU', lambda x: ((x[:, 0] + x[:, 1]) * (x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3])).unsqueeze(-1)]
SIMPLENALU = ['SIMPLENALU', lambda x:  (x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]).unsqueeze(-1)]
LFSR4  = ['LFSR4', lambda x: torch.cat((((x[:, 0] + x[:, 3]) % 2).unsqueeze(-1),  x[:, :-1]), dim=-1)]


set_domain = [[0, 1], [0, 1], [0, 1], [0, 1]]
tensor_list = []
for set_input in set_domain:
    tensor_list.append(torch.FloatTensor(np.random.choice(set_input, size=(10, 1), replace=True)))
X = torch.cat(tensor_list, dim=-1)

X += 0.001


xor = lambda x, y: ((x).type(torch.BoolTensor) ^ (y).type(torch.BoolTensor)).type(torch.FloatTensor)

xor(X, X)


lfsr4 =  lambda x: torch.cat((((x[:, 0] + x[:, 3]) % 2).unsqueeze(-1),  x[:, :-1]), dim=-1)




# (X[:,0] + X[:, 1]) % 2


# lfsr4(X)




def rec(x):
    ok = 0
    for i in range(x.shape[0]):
        if x[i] < 0:
            ok = 1
    if ok == 0:
        return x
    for i in range(x.shape[0]):
        if x[i] < 0:
            x[i] += 1
    return rec(x)


RECURSION = ['RECURSION', rec]


def forloop(x):
    y = x.clone()
    for i in range(x.shape[0]):
        for _ in range(2):
            if y[i] < 2:
                y[i] = y[i] + 2
            else:
                y[i] = y[i] - 2
    return y


FORLOOP = ['FORLOOP', forloop]

# targets for the experiment
SIMPLE_COMPOSED_CONSTANTS = ['SIMPLE_COMPOSED_CONSTANTS', lambda x: (5 * math.pi / 2) * x]
SIMPLE_LINEAR = ['SIMPLE_LINEAR', lambda x: 5 * x + 2]
SIMPLE_QUADRATIC = ['SIMPLE_QUADRATIC', lambda x: x * x + 3 * x]
SIMPLE_SINE_WITH_PHASE = ['SIMPLE_SINE_WITH_PHASE', lambda x: np.sin(3 * x + 2)]
SIMPLE_SINE_MIXTURE = ['SIMPLE_SINE_MIXTURE', lambda x: np.sin(x) + np.sin(3 * x)]
SIMPLE_SINE_MIXTURE_HARD = ['SIMPLE_SINE_MIXTURE_HARD', lambda x: np.sin(2 * x) + np.sin(3 * x)]
SIMPLE_SQUARE_WITH_DIVISION = ['SIMPLE_SQUARE_WITH_DIVISION', lambda x: x * x / (x + 1)]
LOGIC_WITH_LINEAR = ['LOGIC_WITH_LINEAR', lambda x: 3 * x * (x > 0) + x * (x < 0)]
LOGIC_WITH_SQUARE = ['LOGIC_WITH_SQUARE', lambda x: x * x * (x > 0) + x * (x < 0)]
LOGIC_WITH_NEGATION = ['LOGIC_WITH_NEGATION', lambda x: x * (x > 2) + (- x) * (x < 2)]
LOGIC_WITH_SINE = ['LOGIC_WITH_SINE', lambda x: x * (x < 0) + np.sin(2 * x)]

TARGET_SET = [IDENTITY, CONSTANT_MULTIPLY, SQUARE, SINE, BASIC_INEQUALITY, MULTI_INEQUALITY, PROTOTYPING,
              INVERSE, SINE_DIFFERENT_FREQ, SINE_DIFFERENT_FREQ_PHASE_SQUARE, SQUARE_PLUS_CONSTANT_MULTIPLY,
              BASIC_PRODUCT, BASIC_SUM, SINE_SHIFT_OFFSET, SIMPLE_RECURSION, RECURSION, SIGNAL, FORLOOP,
              CONSTANT_INCREASE, CONSTANT_BUILDING,

            # ===========================================
            ### NALU
            # ===========================================
              NALU,
              SIMPLENALU,

            # ===========================================
            ### PHYSICS
            # ===========================================
              GRAVITATION,


           # ===========================================
           ### COMPUTER SCIENCE
           # ===========================================
              SORTING,
              LFSR4,

              # below are the target functions for the experiment
              SIMPLE_COMPOSED_CONSTANTS,
              SIMPLE_LINEAR,
              SIMPLE_QUADRATIC,
              SIMPLE_SINE_WITH_PHASE,
              SIMPLE_SINE_MIXTURE,
              SIMPLE_SINE_MIXTURE_HARD,
              SIMPLE_SQUARE_WITH_DIVISION,
              LOGIC_WITH_LINEAR,
              LOGIC_WITH_SQUARE,
              LOGIC_WITH_NEGATION,
              LOGIC_WITH_SINE
              ]

TARGET_FUNCTIONS = {f[0]: f[1] for f in TARGET_SET}
