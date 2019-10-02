#implementation of approximate TanSig using PLAN as described in article
#Piecewise linear approximation applied to nonlinear function of a neural network
#purpose: to find error level of PLAN when compared to TanSig
import numpy as np
from scipy import arange
import matplotlib.pyplot as plt

def positive(x):
    if np.abs(x) >= 5:
        y = 1
    elif np.abs(x) >= 2.375 and np.abs(x) < 5:
        y = 0.03125*np.abs(x) + 0.84375
    elif np.abs(x)>= 1 and np.abs(x)<2.375:
        y = 0.125*np.abs(x) + 0.625
    #elif np.abs(x) >=0 and np.abs(x) < 1:
        #y = 0.25*np.abs(x) +0.5
    else:
        y = 0.25*np.abs(x) +0.5
    return y
    
print(positive(2))


def plan(x):
    if x >=0:
        y = positive(x)
    else:
        y = -positive(x)
    return y
    
print(plan(-2))

plan = np.vectorize(plan)
x = np.arange(-3.0,3.0,0.1)
plt.plot(x,plan(x))
plt.plot(x,np.tanh(x))
plt.grid(True)
plt.show()

y1 = plan(x)
y2 = np.tanh(x)
#err=mean_squared_error(y2,y1)
err = MSE = np.square(np.subtract(y2,y1)).mean()
print(err)
'''
def plan(x):
    if np.abs(x) >= 5:
        y = 1
    elif np.abs(x) >= 2.375 and np.abs(x) < 5:
        y = 0.03125*np.abs(x) + 0.84375
    elif np.abs(x)>= 1 and np.abs(x)<2.375:
        y = 0.125*np.abs(x) + 0.625
    elif np.abs(x) >=0 and np.abs(x) < 1:
        y = 0.25*np.abs(x) +0.5
    else:
        y = 1-y
    return y
    
print(plan(2))'''