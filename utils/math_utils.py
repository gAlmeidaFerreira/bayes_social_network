import numpy as np

def sigmoid(x, beta, tau):
    return 1 / (1 + np.exp(-beta * (x - tau)))