import numpy as np

def f(x, y): 
    return x**3 + y * x

# Initial condition
x0 = 0
y0 = 1
h = 0.1
y_values = []

# Step 1: Generate first 4 values using Euler or RK4 (you used Euler here)
for i in range(3):  # This gives y1, y2, y3
    y_next = y0 + h * f(x0, y0)
    y_values.append(y_next)
    y0 = y_next
    x0 = x0 + h

# Unpack to named variables
y1, y2, y3 = np.array(y_values).flatten()
x1 = 0.1
x2 = 0.2
x3 = 0.3

j = 0
for j in range(6):
    x4 = x3 + h

    # Milne Predictor
    y_p = y0 + (4 * h / 3) * (2*f(x3, y3) - f(x2, y2) + 2*f(x1, y1))

    # Milne Corrector (iterative)
    for i in range(100):
        y_c = y2 + (h / 3) * (f(x4, y_p) + 4*f(x3, y3) + f(x2, y2))
        if abs(y_c - y_p) < 1e-13:
            print(f"Final Corrected value for x{j+4}:", y_c)
            break
        y_p = y_c

    # Shift values for next step
    y0 = y1
    y1 = y2
    y2 = y3
    y3 = y_c

    x1 = x2
    x2 = x3
    x3 = x4
