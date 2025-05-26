import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10.0         # Length of the domain
T = 2.0          # Total time
n = 101           # Number of spatial points
m = 500           # Number of time steps
alpha = 0.01      # Thermal diffusivity
dx = L / (n - 1)  # Spatial step
dt = T / m        # Time step

# Stability condition
gamma = alpha * dt / dx**2

# Initialize the temperature field
x = np.linspace(0, L, n)  # Spatial grid
u = np.zeros(n)            # Temperature array
u_new = np.zeros(n)        # Array for next time step

# Initial condition (e.g., a sine wave)
u[:] = np.sin(np.pi * x / L)

# Boundary conditions (Dirichlet)
u[0] = 0
u[-1] = 0

# Time-stepping loop (DuFort-Frankel scheme)
for t in range(1, m):
    # Apply DuFort-Frankel scheme to update temperature
    for i in range(1, n-1):
        u_new[i] = u[i] + gamma * (u[i-1] - 2 * u[i] + u[i+1])

    # Update old values
    u[:] = u_new[:]
    
    # Boundary conditions (fixed value)
    u[0] = 0
    u[-1] = 0
    
    # Plot at selected time steps (optional)
    if t % 50 == 0:
        plt.plot(x, u, label=f't = {t*dt:.2f}')
    
# Final plot
plt.xlabel('x')
plt.ylabel('Temperature (u)')
plt.title('DuFort-Frankel Scheme for Heat Equation')
plt.legend()
plt.show()
