import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 41  # number of discretized points
a = 0.5
b = 1.5
dt = 0.0001
t = 0
h = (b - a) / (n - 1)  # space step

# Initialize vectors and matrices
x = np.zeros(n)
u = np.zeros(n)
R = np.zeros(n-2)
v = np.zeros(n-2)
A = np.zeros((n-2, n-2))
B = np.zeros((n-2, n-2))
er = np.zeros(n)
ex = np.zeros(n)

# x array
for i in range(n):
    x[i] = a + (i-1) * h

# Initial condition for u
for i in range(n):
    u[i] = np.sin(np.pi * x[i])

# Calculate the coefficient r
r = dt / (h ** 2)

# Set up the matrix A
A[0, 0] = 2 * (1 + r)
A[0, 1] = -r
for i in range(1, n-3):
    A[i, i] = 2 * (1 + r)
    A[i, i-1] = -r
    A[i, i+1] = -r
A[n-3, n-4] = -r
A[n-3, n-3] = 2 * (1 + r)

# Set up the matrix B
B[0, 0] = 2 * (1 - r)
B[0, 1] = r
for i in range(1, n-3):
    B[i, i] = 2 * (1 - r)
    B[i, i-1] = r
    B[i, i+1] = r
B[n-3, n-4] = r
B[n-3, n-3] = 2 * (1 - r)

# Time-stepping loop
count = 0
while t < 0.25:
    # Step 1: Store u values for the next time step
    v = u[1:n-1]

    # Step 2: Compute the right-hand side vector R
    R = np.dot(B, v)
    R[0] += r * (np.exp(-np.pi**2 * t) * np.sin(np.pi * a) + np.exp(-np.pi**2 * (t + dt)) * np.sin(np.pi * a))
    R[n-3] += r * (np.exp(-np.pi**2 * t) * np.sin(np.pi * b) + np.exp(-np.pi**2 * (t + dt)) * np.sin(np.pi * b))

    # Step 3: Solve for R1 using matrix A
    R1 = np.linalg.solve(A, R)

    # Step 4: Update the solution u
    u[1:n-1] = R1
    u[0] = np.exp(-np.pi**2 * t) * np.sin(np.pi * a)
    u[n-1] = np.exp(-np.pi**2 * t) * np.sin(np.pi * b)

    # Increment time
    t += dt
    count += 1

    # Optional: Display progress every 10 iterations
    if count % 10 == 0:
        print(f"Iteration {count}, Time {t:.4f}")

# Calculate error and exact solution
for i in range(n):
    er[i] = u[i] - np.exp(-np.pi**2 * t) * np.sin(np.pi * x[i])
    ex[i] = np.exp(-np.pi**2 * t) * np.sin(np.pi * x[i])

# Find the maximum error
e1 = np.max(np.abs(er))

# Plot the results
plt.plot(x, ex, label="Exact Solution")
plt.plot(x, u, 'o', label="Numerical Solution")
plt.legend()
plt.show()

