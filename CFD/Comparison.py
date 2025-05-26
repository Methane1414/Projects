import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load C++ simulation data
data = pd.read_csv("turbulent.csv")
x = data["x"]
u_cpp = data["u"]
k_cpp = data["k"]
eps_cpp = data["epsilon"]

# Free-stream velocity
U_infinity = 10.0  # m/s

# Empirical 1/7th power law for velocity profile (Turbulent Boundary Layer Approximation)
def velocity_profile_empirical(y, delta):
    return U_infinity * (y / delta) ** (1 / 7)

# Generate empirical velocity profile
delta = max(x)  # Assume boundary layer thickness is plate length
u_empirical = velocity_profile_empirical(x, delta)

# Plot velocity profile comparison
plt.figure(figsize=(8, 5))
plt.plot(x, u_cpp, label="C++ Simulation (k-ε Model)", color="b", linestyle="-")
plt.plot(x, u_empirical, label="Empirical 1/7th Power Law", color="r", linestyle="--")
plt.xlabel("x (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Comparison of Velocity Profile")
plt.legend()
plt.grid()
plt.show()

# Plot turbulence kinetic energy
plt.figure(figsize=(8, 5))
plt.plot(x, k_cpp, label="Turbulence Kinetic Energy (k)", color="g")
plt.xlabel("x (m)")
plt.ylabel("k (m²/s²)")
plt.title("Turbulence Kinetic Energy Profile")
plt.legend()
plt.grid()
plt.show()

# Plot turbulence dissipation rate
plt.figure(figsize=(8, 5))
plt.plot(x, eps_cpp, label="Turbulence Dissipation Rate (ε)", color="m")
plt.xlabel("x (m)")
plt.ylabel("ε (m²/s³)")
plt.title("Turbulence Dissipation Rate Profile")
plt.legend()
plt.grid()
plt.show()
