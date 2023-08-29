# -*- coding: utf-8 -*-
"""
@author: jorge
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quadrature

def integral(f, a, b, N, metodo):
    h = (b-a)/N
    if metodo =='t':
        intet = (h/2)*np.sum(f[:-1] + f[1:])
        return intet
    if metodo =='s':
        intes = (h/3)*np.sum(f[0:-1:2] + 4*f[1::2] + f[2::2])
        return intes

fint = lambda x: np.sin(x)

a = 0
b = 1*np.pi
N = 10

x = np.linspace(a,b, N+1)
fx = fint(x)

int_1 = integral(fx, a, b, N, 't')
int_2 = integral(fx, a, b, N, 's')
int_3 = quadrature(fint, a, b)

x1 = np.linspace(a,b, 300)
fx1 = fint(x1)

plt.plot(x1,fx1)

for i in range(N):
    xa = [x[i], x[i], x[i+1], x[i+1]]
    ya = [0, fint(x[i]), fint(x[i+1]), 0]
    plt.fill(xa,ya, 'white', edgecolor='black',ls='--')