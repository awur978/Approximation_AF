#implementation of approximate TanSig using PLAN as described in article
#IMPLEMENTING NONLINEAR ACTIVATION FUNCTIONS IN NEURAL NETWORK EMULATORS 
#purpose: to find error level of PLAN when compared to TanSig

#here plenty activation functions were implemented using this method, step,ramp and sigmoid
# only interested in sigmoid and tansig

'''
KEY
t_h = CMP(highlimit,x)
t_l = CMP(x,lowlimit)
t_pn = cmp(x,0)
'''
import numpy as np
from scipy import arange
import matplotlib.pyplot as plt

def cmp(i,j):
    if i >= j:
        y=1
    else:
        y = 0
    return y


#step function
highlimit = 2 # upper limit of saturation points
lowlimit = 0 # lower limit of saturation points
gain = 1/(2*(highlimit)**2) # gradient of the line intersecting the two saturation points

#y = cmp(x,lowlimit)

cmp = np.vectorize(cmp)
x = np.arange(-3.0,3.0,0.1)
#lowlimit for step function should be 0
y_uni = cmp(x,lowlimit) #step function unipolar 
y_bi = 2*(cmp(x,lowlimit))-1 #step function bipolar
plt.plot(x,y_uni)
#plt.plot(x,np.tanh(x))
plt.grid(True)
plt.show()

#RAMP function
def ramp_func(x):
    if x >= highlimit:
        y=1
    elif lowlimit <= x  and lowlimit <highlimit:
        y = 1 + gain*(x-highlimit)
    else:
        y = 0
    return y

def ramp_func2(x):
    t_h = cmp(highlimit,x)
    t_l = cmp(x,lowlimit)
    y = t_l + t_h*t_l*gain*(x-highlimit)
    return y
    
ramp_func2 = np.vectorize(ramp_func2)
x = np.arange(-3.0,3.0,0.1)
y_uni = ramp_func2(x)
plt.plot(x,y_uni)
plt.grid(True)
plt.show()


#SIGMOID

def sig1(x):
    t_h = cmp(highlimit,x)
    t_l = cmp(x,lowlimit)
    t_pn = cmp(x,0)
    
    y = t_pn - ((2*t_pn)-1) * (t_h*t_l*gain*(((2*t_pn-1)*highlimit)-x)**2)
    
    return y
    
sig1 = np.vectorize(sig1)
x = np.arange(-3.0,3.0,0.1)
y_uni = sig1(x)
plt.plot(x,y_uni)
plt.grid(True)
plt.show()

