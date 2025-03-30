import numpy as np
import pandas as pd
from scipy.optimize import minimize
import itertools

# Importing synthetic dataset
data = pd.read_csv("C:\\Users\\chaur\\Documents\\Projects\\CSTR_optimize\\cpp_files\\Sythetic_dataset.csv") 

# Global Constants
R = 8.314  # Universal gas constant
C_A0 = 1.0  # Initial concentration

# Objective function
def objective(params, data):
    k0, Ea = params
    total_error = 0.0
    
    for i in range(len(data)):
        T = data.iloc[i]['T']
        tau = data.iloc[i]['tau']
        X_true = data.iloc[i]['X']
        
        # Calculate rate constant using Arrhenius equation
        k = k0 * np.exp(-Ea / (R * T))
        
        # CSTR equation for conversion
        X_pred = (k * C_A0 * tau) / (1 + k * tau)
        
        # Error calculation (Sum of Squared Errors)
        total_error += (X_pred - X_true) ** 2
    
    return total_error / len(data)

# jacobian gardient function for Newton-CG
def gradient(params, data):
    k0, Ea = params
    grad_k0 = 0.0
    grad_Ea = 0.0

    for i in range(len(data)):
        T = data.iloc[i]['T']
        tau = data.iloc[i]['tau']
        X_true = data.iloc[i]['X']
        
        k = k0 * np.exp(-Ea / (R * T))
        X_pred = (k * C_A0 * tau) / (1 + k * tau)
        
        error = X_pred - X_true

        dk_dk0 = np.exp(-Ea / (R * T))
        dk_dEa = -k0 * (np.exp(-Ea / (R * T)) / (R * T))
        
        dX_dk = (C_A0 * tau) / (1 + k * tau)**2
        
        dX_dk0 = dX_dk * dk_dk0
        dX_dEa = dX_dk * dk_dEa

        grad_k0 += 2 * error * dX_dk0
        grad_Ea += 2 * error * dX_dEa

    grad_k0 /= len(data)
    grad_Ea /= len(data)

    return np.array([grad_k0, grad_Ea])

# Defining a range of initial guesses for k0 and Ea
k0_values = np.linspace(50, 2000, 10)   # 10 values between 50 and 2000
Ea_values = np.linspace(30000, 100000, 10)  # 10 values between 30,000 and 100,000

# Preparing results storage
results = []

for k0_guess, Ea_guess in itertools.product(k0_values, Ea_values):
    initial_guess = [k0_guess, Ea_guess]
    result = minimize(objective, initial_guess, args=(data,), method='Newton-CG', jac=gradient, options={'disp': False})
    
    if result.success:
        k0_opt, Ea_opt = result.x
        final_error = result.fun
        results.append([k0_guess, Ea_guess, k0_opt, Ea_opt, final_error])

# Saving results to CSV
output_file = 'newton_optimization_multiple_results.csv'
results_df = pd.DataFrame(results, columns=['Initial_k0', 'Initial_Ea', 'Optimized_k0', 'Optimized_Ea', 'Error'])
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to '{output_file}")
