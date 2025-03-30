import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import itertools  # For generating combinations


# Importing synthetic dataset
data = pd.read_csv("C:\\Users\\chaur\\Documents\\Projects\\CSTR_optimize\\cpp_files\\Sythetic_dataset.csv") 

# Global Constants
R = 8.314  # Universal gas constant
C_A0 = 1.0  # Initial concentration

# Parameters for Gradient Descent
learning_rate_k0 = 0.1       # Learning rate for k0
learning_rate_Ea = 100        # Higher learning rate for Ea due to its magnitude
epochs = 1000  # Number of iterations

# Initial Values to Iterate Over
initial_k0_values = np.linspace(80, 120, 5)  # 5 values between 80 and 120
initial_Ea_values = np.linspace(40000, 60000, 5)  # 5 values between 40000 and 60000

# Generate all combinations of initial values using itertools.product
initial_combinations = list(itertools.product(initial_k0_values, initial_Ea_values))

# Objective function to calculate Mean Squared Error
def calculate_error(k0, Ea):
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
    
    # Return Mean Squared Error
    return total_error / len(data)

# Gradient calculation
def calculate_gradients(k0, Ea):
    grad_k0 = 0.0
    grad_Ea = 0.0

    for i in range(len(data)):
        T = data.iloc[i]['T']
        tau = data.iloc[i]['tau']
        X_true = data.iloc[i]['X']
        
        # Calculate k using Arrhenius equation
        k = k0 * np.exp(-Ea / (R * T))
        X_pred = (k * C_A0 * tau) / (1 + k * tau)
        
        error = X_pred - X_true
        
        # Derivatives of k with respect to k0 and Ea
        dk_dk0 = np.exp(-Ea / (R * T))
        dk_dEa = -k0 * np.exp(-Ea / (R * T)) / (R * T)
        
        # Derivatives of X_pred with respect to k (using chain rule)
        dX_dk = (C_A0 * tau) / (1 + k * tau)**2
        
        # Applying chain rule to calculate gradients
        dX_dk0 = dX_dk * dk_dk0
        dX_dEa = dX_dk * dk_dEa

        grad_k0 += 2 * error * dX_dk0
        grad_Ea += 2 * error * dX_dEa

    grad_k0 /= len(data)
    grad_Ea /= len(data)

    return grad_k0, grad_Ea

# Prepare results file
output_file = 'gradient_descent_multiple_initials.csv'
header = not os.path.exists(output_file)  # Write header only if file doesn't exist

# Stores results for each initialization
results = []

for k0_init, Ea_init in initial_combinations:  
    
    # Initialize parameters for this run
    k0, Ea = k0_init, Ea_init
    errors = []

    for epoch in range(epochs):
        grad_k0, grad_Ea = calculate_gradients(k0, Ea)
        
        # Updating parameters using gradient descent rule
        k0 -= learning_rate_k0 * grad_k0
        Ea -= learning_rate_Ea * grad_Ea

        # Tracing error every 100 epochs
        if (epoch + 1) % 100 == 0:
            error = calculate_error(k0, Ea)
            errors.append(error)
        
    # Calculating final error and save results
    final_error = calculate_error(k0, Ea)
    results.append([k0_init, Ea_init, k0, Ea, final_error])
    print(f"Initial k0: {k0_init}, Initial Ea: {Ea_init} -> Final k0: {k0:.2f}, Final Ea: {Ea:.2f}, Final Error: {final_error:.6f}")

# Saving results to CSV file
results_df = pd.DataFrame(results, columns=['Initial_k0', 'Initial_Ea', 'Final_k0', 'Final_Ea', 'Final_Error'])
results_df.to_csv(output_file, index=False, mode='a', header=header)
print(f"\nResults saved to '{output_file}'")
