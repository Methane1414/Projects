import numpy as np
import matplotlib.pyplot as plt

n = 4                  # number of elements
h = 1 / n              # element size
x = np.linspace(0, 1, n+1)  # nodes

K = np.zeros((n+1, n+1))  # stiffness matrix
F = np.zeros(n+1)         # load vector

for i in range(n):
    # local stiffness and force (constant f=1)
    K[i][i]     += 1/h
    K[i][i+1]   += -1/h
    K[i+1][i]   += -1/h
    K[i+1][i+1] += 1/h
    
    F[i]     += h/2
    F[i+1]   += h/2

# Apply boundary conditions: u(0) = u(1) = 0
K = K[1:-1, 1:-1]
F = F[1:-1]

u_inner = np.linalg.solve(K, F)
u = np.concatenate(([0], u_inner, [0]))

plt.plot(x, u, 'o-', label='Galerkin FEM')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Simple Galerkin FEM for -u" = 1')
plt.grid(True)
plt.legend()
plt.show()
