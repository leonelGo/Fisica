# Método de Euler atrasado
# Se trabaja con matrices dispersas que ayudan a hacer calculos mas rapido

import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt

Nx = 10
mu = .2

diag = np.zeros(Nx)
infe = np.zeros(Nx-1)
supe = np.zeros(Nx-1)

# condiciones de la matriz

diag[:] = 1 + 2*mu
infe[:] = -mu
supe[:] = -mu

# Condiciones de la frontera

diag[0] = 1
diag[Nx-1] = 1
infe[-1] = 0
supe[0] = 0

A = scipy.sparse.diags(diagonals=[supe, diag, infe], offsets=[1,0,-1],
                       shape=(Nx, Nx), format='csr')

# Aplicamos condiciones iniciales

u0 = u0(x)
u_nm1 = u0


for n in range(0, Nt):
    b = u_nm1

    # Condiciones de frontera
    b[0] = 0
    b[-1] = 0
    # resolver el sistema
    u_n[:] =  scipy.sparse.linalg.spsolve(A, b)
    sol[n]= u_nm1
    