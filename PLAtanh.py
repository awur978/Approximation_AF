#implementation of approximate TanSig using PLAtanh as described in article
#Neural Network based ECG Anomaly Detection on FPGA and Trade-off Analysis
#purpose: to find error level of PLAtanh when compared to TanSig
#MSE = 9.868219573935695e-05
import numpy as np
import matplotlib.pyplot as plt
from scipy import arange
#import sklearn
#from sklearn.metrics import mean_squared_error 

a = 5.5799959
b = 3.02
c = 2.02
d = 1.475
e = 1.125
f = 0.5
g = -0.5
h = -1.125
i = -1.475
j = - 2.02
k = -3.02
l = -5.5799959
def PLAtanh(x):
    if x >= a:
        y = 1.0
    elif x<=a and x>b:
        y = (x/4096.0) + 0.9986377
    elif x<=b and x>c:
        y = (x/32.0) + 0.905
    elif x<=c and x>d:
        y=(x/8.0) + 0.715625
    elif x<=d and x>e:
        y=(x/4.0) + 0.53125
    elif x<=e and x>f:
        y=(x/2.0) + 0.25
    elif x<=f and x>g:
        y=x
    elif x<=g and x >h:
        y=(x/2.0) - 0.25
    elif x<=h and x>i:
        y=(x/4.0) - 0.53125
    elif x<=i and x>j:
        y=(x/8.0) - 0.715625
    elif x<=j and x>k:
        y = (x/32.0) - 0.905
    elif x<=k and x>l:
        y = (x/4096.0) - 0.9986377
    else:
        y = -1.0
    return y
    
PLAtanh = np.vectorize(PLAtanh)
x = np.arange(-3.0,3.0,0.1)
plt.plot(x,PLAtanh(x))
plt.plot(x,np.tanh(x))
plt.grid(True)
plt.show()

y1 = PLAtanh(x)
y2 = np.tanh(x)
#err=mean_squared_error(y2,y1)
err = MSE = np.square(np.subtract(y2,y1)).mean()
print(err)

#y1 = PLAtanh(a)
#print(y1)