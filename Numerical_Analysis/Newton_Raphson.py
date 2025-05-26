import numpy as np
import matplotlib.pyplot as mat
import sympy as sym

x,y = sym.symbols('x,y')
f = x**2+x*y+y**2-7
g = x**3+y**3-9

J = sym.Matrix([[sym.diff(f,x),sym.diff(f,y)],[sym.diff(g,x),sym.diff(g,y)]])

f = sym.lambdify((x, y), f, 'numpy')
g = sym.lambdify((x, y), g, 'numpy')
J_func = sym.lambdify((x, y), J, 'numpy')

def Newton_raphson(x0, y0, max_iteration=1000, tol=1e-6):    
    x,y = x0,y0
    for i in range(max_iteration): 
        initial_vector = np.array([[x],[y]]) 
        F = np.array([[f(x,y)],[g(x,y)]])
        J_inv = np.linalg.inv(J_func(x,y))
        solution_vector = initial_vector - J_inv@F 
        print(solution_vector)
        if np.linalg.norm(solution_vector - initial_vector,ord=2) < tol:
            print("Convergence achieved")
            return solution_vector
        x,y = solution_vector.flatten()
        
x0,y0 = 1.5,0.5
solution = Newton_raphson(x0,y0)

print(f"x = {solution[0,0]:.10f}, y = {solution[1,0]:.10f}")       
    
