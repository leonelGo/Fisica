# Codigo para obtener la solución númerica del atractor de lotentz

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def Sistema(CI, t, p):
    sigma, r, b = p
    x, y, z = CI

    dx = sigma*(y-x)
    dy = r*x - y - x*z
    dz = x*y - b*z

    return [dx, dy, dz]

sigma = 10
r = 28
b = 8/3
p = [sigma, r, b] 


P1 = [0, 2, 0]
P2 = [0, 2.01, 0]
t = np.arange(0, 50, 0.001)

Sol_1 = odeint(Sistema, P1, t, args=(p,))

x1= Sol_1[:, 0]
y1= Sol_1[:, 1]
z1= Sol_1[:, 2]

fig1 = plt.figure(1, figsize=(10,10))
ax= fig1.add_subplot(221, projection = '3d')
ax.plot(x1, y1, z1)



Sol_2 = odeint(Sistema, P2, t, args=(p,))

x2 = Sol_2[:, 0]
y2 = Sol_2[:, 1]
z2 = Sol_2[:, 2]

ax2= fig1.add_subplot(222, projection = '3d')
ax2.plot(x2, y2, z2, color='red')

ax3 = fig1.add_subplot(212)
ax3.plot(t, x1, color = 'red', label= 'Atractor 1')
ax3.plot(t, x2, color = 'black', label= 'Atractor 2')
ax3.set_xlabel('Tiempo')

plt.legend()
plt.show()