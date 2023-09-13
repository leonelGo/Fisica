#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:11:44 2023

@author: Miguel Aguayo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp
import random
#%%
def pendulo(y,t,alpha,beta):
    theta,w= y #¿Condiciones iniciales?
    dydt=[w, -beta*w-alpha*np.sin(theta)] #Sistema de edo
    return dydt
#%% Parametros
g = 9.81
m = 1
L=1
tf = 15
t = np.linspace(0, tf,10**3)


alpha=g/L
#%% Condición inicial
y0 = [np.pi/2,0]  #Varias esta madre para diferentes diagramas de fase
print(y0)
#%%
amortiguamiento=([0,np.sqrt(m*g*L),2*np.sqrt(m*g*L)])    #Parametro de amortiguamiento

labels=['NA','SA','AC']

plt.figure(figsize=(8,8))
l=0
for b in amortiguamiento:
    beta=b/m

####Solución
    sol = odeint(pendulo,y0,t, args=(alpha,beta))

    theta = sol[:,0] #toma la primer columna
    w = sol[:,1]#toma la segunda columna

####Grafica
    plt.subplot(2,1,1)
    plt.plot(t,theta,label=labels[l])
    plt.ylabel(r'$\theta$ [rad]')
    plt.xlabel('Tiempo [s]')
    plt.title('Solución numérica')
    plt.legend()    
    plt.grid(True)


    plt.subplot(2,1,2)
    plt.plot(theta,w,label=labels[l])
    plt.title('Plano de fase')
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'$\omega$ [rad]/[s]')
    plt.legend()
    plt.grid(True)

    l+=1
plt.tight_layout()

plt.savefig('Pendulo_simple.png')
plt.show()
