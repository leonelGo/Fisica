#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:39:42 2023

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random


def edos(y0,t,p):
    m1, m2, k1, k2, b1, b2, L1, L2 = p #Parametros
    x1, v1, x2, v2 = y0 #Condiciones iniciales
    dydt=[v1,
          (-b1*v1-k1*(x1-L1)+k2*(x2-x1-L2))/m1,
          v2,
          (-b2*v2-k2*(x2-x1-L2))/m2]
    return dydt
#%% Parametro ajustables
m1=random.uniform(0, 10)
m2=random.uniform(0, 10)
k1=random.uniform(0, 10)
k2=random.uniform(0, 10)
b1=random.uniform(0, 10)
b2=random.uniform(0, 10)
L1=random.uniform(0, 10)
L2=random.uniform(0, 10)

P=[m1, m2, k1, k2, b1, b2, L1, L2]
print(P)
#%% Condiciones iniciales
x1=1
v1=1/2
x2=3
v2=1/2

y0=np.array([x1, v1, x2, v2])    
#%% Tiempo    
resolucion=10**3
tf=100

t=np.linspace(0,tf,resolucion)


#%%

sol=odeint(edos, y0, t,args=(P,))
x1=sol[:,0]
v1=sol[:,1]
x2=sol[:,2]
v2=sol[:,3]

plt.figure()

plt.subplot(2,2,1)
plt.title('Posición')
plt.plot(t,x1)
plt.plot(t,x2)
plt.ylabel('Posición [m]')
plt.xlabel('Tiempo [s]')

plt.subplot(2,2,2)
plt.title('Posición')
plt.plot(t,v1)
plt.plot(t,v2)
plt.ylabel('Velocidad [ms$^{-1}$]')
plt.xlabel('Tiempo [s]')


plt.subplot(2,1,2)
plt.title('Espacio de fase')
plt.plot(x1,v1)
plt.plot(x2,v2)
plt.xlabel('Distancia [m]')
plt.ylabel('Velocidad [ms$^{-1}$]')


plt.tight_layout()
plt.show