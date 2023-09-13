import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Deficir el sistema de ecuaciónes
def Sistema(CI, t, m, b, k, mu, F, Omega):
    x1, x2, y1, y2 = CI

    M = np.array(m) # Array de las masas
    B1, B2 = 1/M*np.array(b)  # Beta
    A1, A2 = 1/M*np.array(k)  # Alpha
    N1, N2 = 1/M*np.array(mu) # Nu
    G1, G2 = 1/M*np.array(F)
    W1, W2 = Omega

    # Ecuaciones diferenciales
    dx1 = y1
    dx2 = y2
    dy1 = -B1*y1 - A1*x1 + N1*x1**3 + A2*(x2-x1) - N2*(x2-x1)**3 +G1*np.cos(W1*t)
    dy2 = -B2*y2 - A2*(x2-x1) + N2*(x2-x1)**3 + G2*np.cos(W2*t)
    return [dx1, dx2, dy1, dy2]



# Ejercicio 1
m_1 = [1,1]
b_1 = [0,0]
k_1 = [0.4, 1.808]
mu_1 = [-1/6, -1/10]
F_1 = [0, 0]
Omega_1 = [0, 0]
CI = [1, -1/2, 0, 0] # Condiciones Iniciales
t = np.linspace(0, 50, 10000)

Sol_1 = odeint(Sistema, CI, t, args=(m_1, b_1, k_1, mu_1, F_1, Omega_1))

# Graficas ejercicio 1

# Para x1
fig, ax1 = plt.subplots(2, figsize=(7, 7))

# Primer gráfica
ax1[0].plot(t, Sol_1[:, 0], 'r-')
ax1[0].set_xlabel('Tiempo [s]')
ax1[0].set_ylabel('Posición [m]')
ax1[0].set_title(r'Gráfica para $x_{1}$')
ax1[0].grid()

# Segunda gráfica
ax1[1].plot(Sol_1[:, 0], Sol_1[:, 2], 'r-')
ax1[1].set_xlabel('Posición [m]')
ax1[1].set_ylabel('Velocidad [m/s]')
ax1[1].set_title(r'Plano de fases para $x_{1}$')
ax1[1].grid()

fig.tight_layout()
plt.savefig('Ejercicio 1 X1.png')
plt.show(block=False)


# Para x2

fig2, ax2 = plt.subplots(2, figsize=(7, 7))

ax2[0].plot(t, Sol_1[:, 1], 'b-')
ax2[0].set_xlabel('Tiempo [s]')
ax2[0].set_ylabel('Posición [m]')
ax2[0].set_title(r'Gráfica para $x_{2}$')
ax2[0].grid()


ax2[1].plot(Sol_1[:, 1], Sol_1[:, 3], 'b-')
ax2[1].set_xlabel('Posición [m]')
ax2[1].set_ylabel('Velocidad [m/s]')
ax2[1].set_title(r'Plano de fases para $x_{2}$')
ax2[1].grid()

fig2.tight_layout()
plt.savefig('Ejercicio 1 X2.png')
plt.show(block=False)


# Ejercicio 2

m_2 = [1,1]
b_2 = [1/10,1/5]
k_2 = [2/5, 1]
mu_2 = [1/6, 1/10]
F_2 = [1/3, 1/5]
Omega_2 = [1, 3/5]
CI_2 = [0.7, 0.1, 0, 0] # Condiciones Iniciales
t = np.linspace(0, 150, 10000)

Sol_2 = odeint(Sistema, CI_2, t, args=(m_2, b_2, k_2, mu_2, F_2, Omega_2))

fig3, ax3 = plt.subplots(2, figsize=(7, 7))

# Para x1
ax3[0].plot(t, Sol_2[:, 0], 'r-')
ax3[0].set_xlabel('Tiempo [s]')
ax3[0].set_ylabel('Posición [m]')
ax3[0].set_title(r'Gráfica para $x_{1}$')
ax3[0].grid()


ax3[1].plot(Sol_2[:, 0], Sol_2[:, 2], 'r-')
ax3[1].set_xlabel('Posición [m]')
ax3[1].set_ylabel('Velocidad [m/s]')
ax3[1].set_title(r'Plano de fases para $x_{1}$')
ax3[1].grid()

fig3.tight_layout()
plt.savefig('Ejercicio 2 X1.png.png')
plt.show(block=False)


# Para x2

fig4, ax4 = plt.subplots(2, figsize=(7, 7))

ax4[0].plot(t, Sol_2[:, 1], 'b-')
ax4[0].set_xlabel('Tiempo [s]')
ax4[0].set_ylabel('Posición [m]')
ax4[0].set_title(r'Gráfica para $x_{2}$')
ax4[0].grid()


ax4[1].plot(Sol_2[:, 1], Sol_2[:, 3], 'b-')
ax4[1].set_xlabel('Posición [m]')
ax4[1].set_ylabel('Velocidad [m/s]')
ax4[1].set_title(r'Plano de fases para $x_{2}$')
ax4[1].grid()

fig4.tight_layout()
plt.savefig('Ejercicio 2 X2.png.png')
plt.show()