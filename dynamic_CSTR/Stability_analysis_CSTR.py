import numpy as np
from scipy.optimize import approx_fprime
from scipy.linalg import eigvals

# System parameters
F0, T0, CA0 = 40.0, 530, 0.5  
alpha, E, R = 7.08e10, 30000, 1.99  
rho, CP, U, AH = 50, 0.75, 150, 250  
rhoJ, VJ, CJ, TJ0 = 62.3, 48.0, 1.0, 530  
Kc = 4  
EPSILON = 1e-6  

# Steady-state values (from CSV)
V_SS, Ca_SS, T_SS, Tj_SS = 48.0, 0.245016, 600.001, 594.62  

# Compute reaction rate at steady state
k_SS = alpha * np.exp(-E / (R * T_SS))

def CSTR_system(X):
    V, Ca, T, Tj = X  
    k = alpha * np.exp(-E / (R * T))  
    FJ = 49.9 - Kc * (600 - T)  

    dV = F0 - (10 * V - 440)
    dCa = (F0 * CA0 - (10 * V - 440) * Ca - k * V * Ca) / V
    dT = (F0 * T0 - (10 * V - 440) * T + (30000 * V * k * Ca - U * AH * (T - Tj))) / (rho * CP * V)
    dTj = (FJ / VJ) * (TJ0 - Tj) + (U * AH * (T - Tj)) / (rhoJ * VJ * CJ)
    
    return np.array([dV, dCa, dT, dTj])

X_SS = np.array([V_SS, Ca_SS, T_SS, Tj_SS])
epsilon = np.sqrt(np.finfo(float).eps)  
J = approx_fprime(X_SS, CSTR_system, epsilon)  

eigenvalues = eigvals(J)  

print("\nJacobian Matrix at Steady State:\n", J)
print("\nEigenvalues of the Jacobian:\n", eigenvalues)

if np.all(np.real(eigenvalues) < 0):
    print("\nThe system is STABLE.")
elif np.any(np.real(eigenvalues) > 0):
    print("\nThe system is UNSTABLE.")
else:
    print("\nThe system is on the verge of instability.")
