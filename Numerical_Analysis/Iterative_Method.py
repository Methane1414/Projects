import numpy as np

def f1(x, y):
    return x**3 + y**2 - 4

def f2(x, y):
    return x * y - 1

def jacobian(x, y):
    J = np.array([[2*x, 2*y],
                  [y, x]])
    return J

x0 = 1.0  # initial guess for x
y0 = 10  # initial guess for y

tol = 1e-10
max_iter = 100

for i in range(max_iter):
    F = np.array([f1(x0, y0), f2(x0, y0)])
    J = jacobian(x0, y0)
    
    # Solve J * delta = -F
    try:
        delta = np.linalg.solve(J, -F)
    except np.linalg.LinAlgError:
        print("Jacobian is singular.")
        break

    x0 += delta[0]
    y0 += delta[1]
    
    if np.linalg.norm(delta, ord=2) < tol:
        print(f"Converged in {i+1} iterations.")
        break

print("Solution:")
print(f"x = {x0}")
print(f"y = {y0}")
