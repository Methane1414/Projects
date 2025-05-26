import numpy as np

# Define the differential equation dy/dx = f(x, y)
def f(x, y): 
    f = x**3 + y * x
    return f

# Initialize storage for the first few y-values
y_values = []

# Initial conditions
x0 = 0
y0 = 1
h = 0.1  # Step size

# Generate y1, y2, y3 using simple Euler's method
for i in range(3):
    y_next = y0 + h * f(x0, y0)  # Euler step
    y_values.append(y_next)      # Store the computed y value
    y0 = y_next                  # Update y0 for next iteration
    x0 = x0 + h                  # Move to next x value

# Unpack initial values for Adams-Bashforth-Moulton method
y1, y2, y3 = np.array(y_values).flatten()

# Predictor-Corrector loop (6 additional steps)
for j in range(6):  
    # Predictor: Adams-Bashforth 4-step explicit formula
    y_p = y3 + (h / 24) * (
        55 * f(x0 + (j + 3) * h, y3) 
        - 59 * f(x0 + (j + 2) * h, y2) 
        + 37 * f(x0 + (j + 1) * h, y1) 
        - 9  * f(x0 + j * h, y0)
    )
    
    # Corrector loop: Adams-Moulton 4-step implicit formula
    for i in range(100):    
        y_c = y3 + (h / 24) * (
            9  * f(x0 + (j + 4) * h, y_p) 
            + 19 * f(x0 + (j + 3) * h, y3) 
            - 5  * f(x0 + (j + 2) * h, y2) 
            +      f(x0 + (j + 1) * h, y1)
        )
        
        # Print intermediate corrected value
        #print(y_c)
        
        # Convergence check
        if abs(y_c - y_p) < 1e-13:
            print(f"Final Corrected value for x{j + 4}:", y_c)
            break
        
        # Update predictor with corrected value
        y_p = y_c

    # Shift y-values for next step (advance the solution window)
    y3 = y_c
    y2 = y3
    y1 = y2
    y0 = y1
