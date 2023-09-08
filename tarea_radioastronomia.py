"""tarea radioastronomia."""
import math
import numpy as np
from scipy import constants as C
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 8)


# Variables
Temp = [35000, 10000, 8000, 6000, 5000, 3000, 2000]  # Kelvin
lamb = []  # metros

# Ley de Wien
for i in Temp:
    lamb += [C.Wien/i]


# Ley de planck
x = np.linspace(.01*10**-6, 100*(10**-6), 100000)  # metros

C1 = 2*C.h*(C.c**2)  # J m^2/s
C2 = C.h*C.c/C.Boltzmann  # m/K


def intensidad(longitud, temp):
    """Funcion de la intensidad de un cuerpo negro."""
    cte = C2/(longitud*temp)
    return (C1/((longitud**5)*(math.e**(cte)-1)))*10**-6  # W/m^2 \mu m sr^-1


fig, eje = plt.subplots()


M = 0
while M < len(Temp):
    k = intensidad(x, Temp[M])
    eje.plot(x*10**6, k, label=str(Temp[M]) + " K", lw=2)
    eje.plot(lamb[M]*10**6*np.ones(2), np.linspace(0, max(k), 2), '.k')
    B = str(int(lamb[M]*10**9))
    eje.annotate(B + ' nm', (lamb[M]*10**6, max(k)+10**(9-M)))
    M += 1

eje.plot(0.38*np.ones(2), np.linspace(0, 10**13, 2),
         linestyle=(1, (4, 10)), color='#F08080')
eje.plot(0.75*np.ones(2), np.linspace(0, 10**13, 2),
         linestyle=(1, (4, 10)), color='#F08080')

plt.xscale("log")
plt.yscale("symlog")
eje.grid("both")
eje.set_ylim([10**1, 10**13])
eje.set_xlim([10**-2, 100])


eje.set_title("Espectro de cuerpo negro")
eje.text(0.41, 10**12, 'Visible', fontsize=12, color='blue')
eje.text(0.2, 10**12, 'UV', fontsize=12, color='blue')
eje.text(1.1, 10**12, 'IR', fontsize=12, color='blue')
eje.set_xlabel(r"Longitud de onda [$\mu$m]")
eje.set_ylabel(r"Intencidad [W/m$^2$ $\mu$m sr$^{-1}$]")
eje.legend()

plt.savefig('Intensidad de cuerpo negro.pdf')
plt.show()
