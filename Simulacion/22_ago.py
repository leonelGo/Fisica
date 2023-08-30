import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def potencial(a, V, zi, zf):
    z = np.linspace(zi, zf, 100)
    k = lambda z: 2*a/(np.sqrt(z**2 + 4*a**2))
    kz = k(z)
    Phi = np.zeros(len(z))
    for i in range(0, len(z)):
        K = lambda phi: 1/np.sqrt(1-kz[i]**2*(np.sin(phi)**2))
        phi2 = integrate.quadrature(K, 0, np.pi/2)
        Phi[i] = (V/2) * (1-((kz[i]*z[i])*z[i])/(np.pi*a))
    return Phi, z

al = range(1,4)
V=1
plt.figure(1)
for a in al:
    Phi, z = potencial(a, V, 0.05, 10)
    plt.plot(z, Phi, label= f'a = {a}')
    plt.legend()

plt.ylabel(' Potencial, $\Phi (z)')
plt.xlabel('Altura, z')
