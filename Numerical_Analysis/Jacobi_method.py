import numpy as np

def sor(A, b, w, tol=1e-6, max_iter=100):
    """
    Successive Over-Relaxation (SOR) Method to solve Ax = b
    Parameters:
        A : numpy array : Coefficient matrix
        b : numpy array : Right-hand side vector
        w : float : Relaxation factor (0 < w < 2)
        tol : float : Convergence tolerance
        max_iter : int : Maximum iterations
    Returns:
        x : numpy array : Solution vector
    """
    n = len(b)
    x = np.zeros(n)  # Initial guess (zero vector)
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (1 - w) * x[i] + (w / A[i, i]) * (b[i] - sigma)
        
        # Check for convergence using infinity norm
        if np.linalg.norm(x - x_old, np.inf) < tol:
            print(f'SOR converged in {k+1} iterations.')
            return x
    
    print(f'SOR did not converge in {max_iter} iterations.')
    return x

# Example Usage
A = np.array([[4, -1, 0], 
              [-1, 4, -1], 
              [0, -1, 4]], dtype=float)  # Coefficient matrix

b = np.array([15, 10, 10], dtype=float)  # Right-hand side vector
w = 1.25  # Relaxation factor (1.0 is Gauss-Seidel, 1.25 is typical SOR)

x = sor(A, b, w)
print("Solution:", x)
