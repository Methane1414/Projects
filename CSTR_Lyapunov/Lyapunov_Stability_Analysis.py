import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigvals

# Chemical Engineering Context: CSTR Stability Analysis

# State-space matrix (linearized CSTR system)
# x1 = concentration deviation
# x2 = temperature deviation
A = np.array([
    [-1.2,  0.5],   # Effect of concentration & temperature on concentration
    [-0.7, -1.5]    # Effect of concentration & temperature on temperature
])

# Lyapunov candidate matrix (must be positive definite)
P = np.array([
    [2, 0],  # Strong weight on concentration stability
    [0, 3]   # Strong weight on temperature stability
])

# Compute Lyapunov equation result: A^T P + P A
Lyapunov_matrix = A.T @ P + P @ A

# Function to check negative definiteness
def is_negative_definite(matrix):
    """Check if the matrix is negative definite (all eigenvalues negative)."""
    eigenvalues = eigvals(matrix)
    return np.all(eigenvalues < 0), eigenvalues

# Check stability
stable, eigenvalues = is_negative_definite(Lyapunov_matrix)

# Print results
print("Lyapunov Matrix (A^T P + P A):\n", Lyapunov_matrix)
print("Eigenvalues:", eigenvalues)
if stable:
    print("The system is STABLE: The temperature & concentration deviations will decay over time.")
else:
    print("The system is UNSTABLE: The temperature & concentration deviations will grow over time!")

# -- Visualization 1: Heatmap of Lyapunov Matrix --
plt.figure(figsize=(5, 4))
sns.heatmap(Lyapunov_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Lyapunov Matrix (A^T P + P A)")
plt.show()

# -- Visualization 2: Eigenvalues Plot --
plt.figure(figsize=(5, 4))
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.scatter(range(len(eigenvalues)), eigenvalues, color="red", marker="o", label="Eigenvalues")
plt.title("Eigenvalues of Lyapunov Matrix")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()

# ---Visualization 3: Phase Portrait (System Dynamics) ---
x1_range = np.linspace(-2, 2, 20)
x2_range = np.linspace(-2, 2, 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Compute dx/dt = A * x for each point in the grid
DX1 = A[0, 0] * X1 + A[0, 1] * X2
DX2 = A[1, 0] * X1 + A[1, 1] * X2

plt.figure(figsize=(6, 6))
plt.quiver(X1, X2, DX1, DX2, color="blue", alpha=0.7)
plt.xlabel("Reactant Concentration Deviation (x1)")
plt.ylabel("Temperature Deviation (x2)")
plt.title("CSTR System Phase Portrait (Vector Field)")
plt.grid()
plt.show()
