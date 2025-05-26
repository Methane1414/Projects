import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10.0        # Length of the domain
T = 10.0         # Total time
n = 101          # Number of spatial points
m = 500          # Number of time steps
c = 1.0          # Wave speed
dx = L / (n - 1) # Spatial step size
dt = T / m       # Time step size

# Stability condition (CFL condition)
if c * dt / dx > 1:
    print("Warning: The scheme may be unstable, adjust dt or dx.")

# Initialize the wave function
x = np.linspace(0, L, n)  # Spatial grid
u = np.zeros(n)            # Wave function array
u_new = np.zeros(n)        # Array for next time step

# Initial condition: a Gaussian pulse
u[:] = np.exp(-(x - L/2)**2)

# Boundary conditions (fixed)

u[0] = 0
u[-1] = 0

# Time-stepping loop (Explicit scheme)
for t in range(1, m):
    # Apply the explicit scheme (Lax-Wendroff)
    for i in range(1, n-1):
        u_new[i] = u[i] - (c * dt / (2 * dx)) * (u[i+1] - u[i-1])

    # Update old values for next iteration
    u[:] = u_new[:]
    
    # Boundary conditions (fixed)
    u[0] = 0
    u[-1] = 0
    
    # Plot at selected time steps (optional)
    if t % 50 == 0:
        plt.plot(x, u, label=f't = {t*dt:.2f}')

# Final plot
plt.xlabel('x')
plt.ylabel('Wave Function (u)')
plt.title('1D Wave Equation using Explicit Scheme')
plt.legend()
plt.show()
