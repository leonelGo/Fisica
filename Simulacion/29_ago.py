import numpy as np
import matplotlib.pyplot as plt


def euler(f, y0, t):
    y = [y0]
    for i in np.arange(0, len(t)-1):
        h = t[i+1]-t[i]
        y.append(y[i]+f(y[i],t[i])*h)


    return y

def f(y,t):
    dydt= -y*np.cos(t)
    return dydt


y0=1/2
h=.1
t=np.arange(0 ,5+h, h)

y= euler(f, y0, t)

#Analitica
y_a=1/2*np.exp(-np.sin(t))

plt.plot(t,y, label='Metodo Euler')
plt.plot(t,y_a , label='Sol. analitica')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()