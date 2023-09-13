import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Ecuacion diferencial
def pendulo(y, t, beta , alpha ):
    theta , omega = y
    dydt = [omega , 
            -beta *omega -alpha*np.sin(theta )]
    return dydt

# Definiendo las constantes
l = 2
m = 3.4
b = 10
g = 9.81
alpha = np.sqrt(g/l)
beta = np.array([0, 2*alpha, np.sqrt(b/m)])

# Definiendo el tiempo y condiciones iniciales
t = np.arange(0, 10*np.pi, 0.01)
y0 = [np.pi/2, 0]

# Pendulo simple, Am. critico, Am Subamortiguado
Name = ['$\\beta = 0$', '$\\beta = 2\\alpha$', '$\\beta < 2\\alpha$'] 


# Graficas
fig, (ax, ax2) = plt.subplots(2, figsize=(7, 7))


for i in range(len(beta)):
    sol = odeint(pendulo, y0, t, args=(beta[i], alpha))
    ax.plot(t, sol[:, 0], label=Name[i])
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel(r'$\theta$ [rad]')
    ax.grid()

    ax2.plot(sol[:, 0], sol[:, 1], label=Name[i])
    ax2.grid()
    ax2.set_xlabel(r'$\theta$ [rad]')
    ax2.set_ylabel(r'$\omega$ [rad/s]')

ax.legend()
ax2.legend()

plt.tight_layout()
plt.savefig('Solucion pendulo amortiguado.png')
plt.show()










