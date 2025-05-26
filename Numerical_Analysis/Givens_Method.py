import numpy as np

# Input symmetric matrix
A = np.array([[4, -2, 2],
              [-2, 2, -4],
              [2, -4, 3]], dtype=float)

n = len(A)
T = A.copy()

# --- Step 1: Givens rotations to tridiagonal form ---
for j in range(n - 2):  # Loop over column
    for i in range(n - 1, j, -1):  # Eliminate below subdiagonal
        a = T[i - 1][j]
        b = T[i][j]
        if abs(b) < 1e-12:
            continue
        r = np.sqrt(a**2 + b**2)
        c = a / r
        s = -b / r

        # Apply rotation to rows i-1 and i
        for k in range(n):
            tau1 = T[i - 1][k]
            tau2 = T[i][k]
            T[i - 1][k] = c * tau1 - s * tau2
            T[i][k]     = s * tau1 + c * tau2

        # Apply rotation to columns i-1 and i
        for k in range(n):
            tau1 = T[k][i - 1]
            tau2 = T[k][i]
            T[k][i - 1] = c * tau1 - s * tau2
            T[k][i]     = s * tau1 + c * tau2

# Zero small entries to emphasize tridiagonal form
for i in range(n):
    for j in range(n):
        if abs(i - j) > 1 and abs(T[i][j]) < 1e-10:
            T[i][j] = 0.0

print("Tridiagonal matrix:")
print(np.round(T, 6))

# --- Step 2: Sturm sequence count for λ ---
def sturm_count(T, lam):
    n = len(T)
    p = np.zeros(n)
    p[0] = T[0][0] - lam
    if p[0] == 0:
        p[0] = -1e-14
    count = 0
    if p[0] < 0:
        count += 1

    for i in range(1, n):
        a = T[i][i] - lam
        b = T[i][i - 1]
        p[i] = a * p[i - 1] - b**2 * (p[i - 2] if i > 1 else 1)
        if p[i] == 0:
            p[i] = -1e-14
        if p[i] * p[i - 1] < 0:
            count += 1

    return count

# --- Step 3: Bisection to find eigenvalues ---
def find_eigenvalues(T, lower, upper, tol=1e-10):
    eigenvalues = []
    for k in range(n):
        a = lower
        b = upper
        while b - a > tol:
            m = (a + b) / 2
            count = sturm_count(T, m)
            if count <= k:
                a = m
            else:
                b = m
        eigenvalues.append((a + b) / 2)
    return eigenvalues

# --- Step 4: Solve and print ---
eigvals = find_eigenvalues(T, lower=-10, upper=10)

print("\nEigenvalues:")
for i, val in enumerate(eigvals):
    print(f"Eigenvalue {i+1} =", val)
