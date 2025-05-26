import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 50                     # Number of spatial points
a, b = 0.0, 1.0            # Domain
dx = (b - a) / (n - 1)
x = np.linspace(a, b, n)

dt = 0.001                 # Time step
t_final = 0.1              # Final time
r = dt / dx**2

# Initial condition: piecewise linear
u = np.zeros(n)
u[:n//2] = 1 + 2 * x[:n//2]
u[n//2:] = 3 - 2 * x[n//2:]

# Construct the coefficient matrix A (tridiagonal)
main_diag = (1 + 2 * r) * np.ones(n - 2)
off_diag = -r * np.ones(n - 3)
A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)

# Time stepping
t = 0
while t < t_final:
    t += dt
    rhs = u[1:-1]  # Internal nodes only
    u_inner = np.linalg.solve(A, rhs)
    u[1:-1] = u_inner  # Update solution (BCs remain zero)

# Plot result
plt.plot(x, u, label='Backward Euler')
plt.title("1D Heat Equation (Backward Euler)")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.grid(True)
plt.legend()
plt.show()
