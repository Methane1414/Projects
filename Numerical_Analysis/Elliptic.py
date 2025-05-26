import numpy as np
import matplotlib.pyplot as plt

N = 100 # grid size
u1 = np.zeros((N, N))  # standard 5-point
u2 = np.zeros((N, N))  # diagonal 5-point
max_iter = 500

# Apply non-zero boundary condition on top edge
u1[:, -1] = 100
u2[:, -1] = 100

# Standard 5-point method
for _ in range(max_iter):
    for i in range(1, N-1):
        for j in range(1, N-1):
            u1[i, j] = 0.25 * (u1[i+1, j] + u1[i-1, j] + u1[i, j+1] + u1[i, j-1])

# Diagonal (9-point) method
for _ in range(max_iter):
    for i in range(1, N-1):
        for j in range(1, N-1):
            u2[i, j] = (1/20) * (
                4*(u2[i+1, j] + u2[i-1, j] + u2[i, j+1] + u2[i, j-1]) +
                (u2[i+1, j+1] + u2[i+1, j-1] + u2[i-1, j+1] + u2[i-1, j-1])
            )

# Plot both
plt.subplot(1, 2, 1)
plt.imshow(u1, cmap='hot')
plt.title("Standard 5-Point")

plt.subplot(1, 2, 2)
plt.imshow(u2, cmap='hot')
plt.title("Diagonal 5-Point")

plt.show()
