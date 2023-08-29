import numpy as np
#import scipy.integrate as integrate

a=3
print(a)

def dy(y, t):
    dy= -y*np.cos(t)
    return dy

t=np.arange(0 ,5, 0.1)
print(t)