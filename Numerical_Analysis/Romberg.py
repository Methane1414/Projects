import numpy as np

def f(x):
    return np.sin(x)-4*x**7+5*np.sqrt(45*x**5)

a, b = 0, np.pi
n = 4
R = np.zeros((n, n))

for i in range(n):
    N = 2**i
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    R[i, 0] = h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])

for j in range(1, n):
    for i in range(j, n):
        R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

# Set printing options for consistent formatting
np.set_printoptions(precision=10, floatmode='fixed', suppress=False)

print("Romberg Integration Table:")
print(R)

print("\nApproximated integral =", R[n-1, n-1])
