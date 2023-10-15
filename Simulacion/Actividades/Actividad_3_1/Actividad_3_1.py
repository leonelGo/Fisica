import numpy as np
import matplotlib.pyplot as plt

# Método de Euler
def euler(Nt ,dx ,dt ,u0 ,L ,alpha):
    
    mu     = alpha*(dt/(dx**2))    

    x      = np.arange(0, L + dx, dx)
    t      = np.arange(0, Nt*dt, dt)    
    sol    = np.zeros((Nt, len(x)))
    u_n    = np.zeros(len(x))    

    u0 = u0(x)    
    u = u0    

    for n in range(Nt):
        u[0]   = 100
        u[-1]  = 0   
        u_n[1:-1]   = u[1:-1] + mu*(u[:-2] - 2*u[1:-1] + u[2:])
        u = u_n
        sol[n] = u
    return x, t, mu, sol
        


# Condiciones del problema

L = 1 
Nt = 5000
dx = 0.05
dt = 0.00004
alpha = 1
u0 = lambda x: np.zeros(len(x)) 

# Solución
x, t, mu ,sol = euler(Nt, dx, dt, u0, L, alpha)


# Gráfica

plt.figure(figsize=(8 ,7))

plt.contourf( x ,t , sol )
plt.ylabel('Tiempo [s]')
plt.xlabel('Posición [m]')
barra = plt.colorbar()
barra.set_label('Temperatura [°C]')

plt.savefig('Grafica de temperatura.png')
plt.show()

