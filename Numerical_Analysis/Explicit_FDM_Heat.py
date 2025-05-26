import numpy as np
import matplotlib.pyplot as plt

n = 6                     # number of spatial grid points
a, b = 0, 1
dt = 0.02                 # time step
t_final = 2 * dt          # run for 2 time steps
h = (b - a) / (n - 1)
x = np.linspace(a, b, n)

# Initial condition (piecewise linear)
u = np.zeros(n)
u[:n//2] = 1 + 2 * x[:n//2]
u[n//2:] = 3 - 2 * x[n//2:]

r = dt / h**2

# Construct A matrix (implicit time integration)
A = np.diag((1 - 2*r) * np.ones(n - 2)) + \
    np.diag(r * np.ones(n - 3), 1) + \
    np.diag(r * np.ones(n - 3), -1)

# Time stepping
t = 0
while t < t_final:
    t += dt
    v = u[1:-1]           # interior values
    R = A @ v
    R[0] += r * u[0]      # apply BC on left
    R[-1] += r * u[-1]    # apply BC on right
    u[1:-1] = R           # update solution (Dirichlet BCs: u[0], u[-1] stay 0)

# Plot result
plt.plot(x, u, 'o-', label='u(x,t)')
plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Heat Equation (Explicit, 2 Steps)")
plt.grid(True)
plt.legend()
plt.show()
