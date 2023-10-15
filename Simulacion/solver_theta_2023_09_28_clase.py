# -*- coding: utf-8 -*-
"""
@author: Jorge M. Montes A.
Simulación de Procesos Físicos, Licenciatura en Física
Universidad de Guadalajara, CUCEI
"""
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solver(I, alpha, f, L, Nx, Nt,dx,dt, theta, u_L, u_R):
    """
    Theta = 0: Euler adelantado
    Theta = 1: Euler atrasado
    Theta = 1/2: Crank-Nicolson
    """
    x = np.linspace(0, L, Nx+1)
    T = Nt*dt
    t = np.linspace(0, T, Nt+1)
    mu = alpha*(dt/dx**2)
    
    print('dt=%g' % dt)
    print('dx=%g' % dx)
    print('mu=%g' % mu)
    
    if isinstance(alpha, (float,int)):
        alpha = np.zeros(Nx+1) + alpha   
    # Condiciones de frontera constantes    
    if isinstance(u_L, (float,int)):
        u_L_ = float(u_L)  
        u_L = lambda t: u_L_
    if isinstance(u_R, (float,int)):
        u_R_ = float(u_R)
        u_R = lambda t: u_R_
    
    # Funciones para las CI y el forzamiento
    if f is None or f == 0:
        f = lambda x, t: 0
    if I is None or I == 0:
        I = lambda x: 0    
        
    u   = np.zeros(Nx+1)   
    u_n = np.zeros(Nx+1)   

    mul = 0.5*mu*theta
    mur = 0.5*mu*(1-theta)

    diagonal = np.zeros(Nx+1)
    lower    = np.zeros(Nx)
    upper    = np.zeros(Nx)
    b        = np.zeros(Nx+1)
    # Generamos las diagonales
    diagonal[1:-1] = 1 + mul*(alpha[2:] + 2*alpha[1:-1] + alpha[:-2])
    lower[:-1] = -mul*(alpha[1:-1] + alpha[:-2])
    upper[1:]  = -mul*(alpha[2:] + alpha[1:-1])
    # condiciones de frontera
    diagonal[0] = 1
    upper[0] = 0
    diagonal[Nx] = 1
    lower[-1] = 0
    # Armamos la matriz dispersa
    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1],
        shape=(Nx+1, Nx+1),
        format='csr')
    #Aplicamos condiciones iniciales
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])
    # generamos un espacio vacio para la solucion    
    sol = np.zeros((Nt,len(u)))
    # resolvemos para cada paso de tiempo
    for n in range(0, Nt):
        b[1:-1] = u_n[1:-1] + mur*(
            (alpha[2:] + alpha[1:-1])*(u_n[2:] - u_n[1:-1]) -
            (alpha[1:-1] + alpha[0:-2])*(u_n[1:-1] - u_n[:-2])) + \
            dt*theta*f(x[1:-1], t[n+1]) + \
            dt*(1-theta)*f(x[1:-1], t[n])
        # Condiciones de frontera
        b[0]  = u_L(t[n+1])
        b[-1] = u_R(t[n+1])
        u[:] = scipy.sparse.linalg.spsolve(A, b)
        u_n[:] = u[:]
        sol[n] = u[:]

        
    return u, x, t, sol


alpha = 1
f = lambda x,t: 100*np.cos(30*t) # funcion de forzamiento
L = 1
Nx = 50
Nt = 20
dt = 0.005
dx = L/Nx
u_L = 0 # lambda t: 0.5*np.cos(t)
u_R = 0 # lambda t: 0.5*np.cos(t)

# nuevo codigo para la actividad 3_2

a1 = 1
a2 = a1/2
a3 = a2/2


#

I = lambda x: np.piecewise(x, [x <= 0.5, x > 0.5],
                            [lambda a: 2*a,
                             lambda a: 2-2*a ])

u, x, t, sol = solver(I, alpha, f, L, Nx, Nt, dx, dt, theta=1, u_L=u_L, u_R=u_R)
u2, x2, t2, sol2 = solver(I, alpha, f, L,  Nx, Nt, dx, dt, theta=1/2, u_L=u_L, u_R=u_R)

fig1 = plt.figure(11)
ax = fig1.add_subplot(211)
ax.contourf(x, t[:-1], sol, cmap='viridis')
ax.set_ylabel('tiempo')


ax2 = fig1.add_subplot(212)
ax2.contourf(x2, t2[:-1], sol2, cmap='viridis')
ax2.set_ylabel('tiempo')

fig3 = plt.figure(30)
ax1 = fig3.add_subplot(111)

lns = []

for i in range(len(sol)):
    l1, = ax1.plot(x, sol[i,:], '-k')
    lns.append([l1])
plt.tight_layout()
ani = animation.ArtistAnimation(fig3, lns, interval=1)


plt.show()
