import numpy as np

A = np.array([[4, -2, 2],
              [-2, 2, -4],
              [2, -4, 3]], dtype=float)

n = len(A)
D = A.copy()
max_iter = 100
tol = 1e-10

for k in range(max_iter):
    # Find largest off-diagonal element in D
    max_val = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if abs(D[i][j]) > abs(max_val):
                max_val = D[i][j]
                p = i
                q = j
    
    # Convergence check
    if abs(max_val) < tol:
        break
    
    # Compute rotation angle
    if D[p][p] == D[q][q]:
        theta = np.pi / 4
    else:
        theta = 0.5 * np.arctan(2 * D[p][q] / (D[q][q] - D[p][p]))
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    # Create rotation matrix J (only p and q rows/cols change)
    J = np.eye(n)
    J[p][p] = cos
    J[q][q] = cos
    J[p][q] = sin
    J[q][p] = -sin
    
    # Apply similarity transformation D = Jᵀ D J
    D = J.T @ D @ J

# Eigenvalues are diagonal of D
for i in range(n):
    print(f"Eigenvalue {i+1} =", D[i][i])
