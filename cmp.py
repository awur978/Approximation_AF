#implementation of approximate TanSig using PLAN as described in article
#IMPLEMENTING NONLINEAR ACTIVATION FUNCTIONS IN NEURAL NETWORK EMULATORS 
#purpose: to find error level of PLAN when compared to TanSig

#here plenty activation functions were implemented using this method, step,ramp and sigmoid
# only interested in sigmoid and tansig

import numpy as np
def cmp(i,j):
    if i >= j:
        y=1
    else:
        y = 0
    return y

#step function
highlimit = 2 # upper limit of saturation points
lowlimit = -2 # lower limit of saturation points