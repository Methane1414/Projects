import numpy as np
import scipy.integrate as spi
import pandas as pd
from numpy.polynomial.legendre import leggauss

def aggregation_kernel(x, y, k=1.0, alpha=1.0):
    """
    Defines the aggregation kernel β(x, y) = k (x^α + y^α).
    Prevents invalid values by ensuring x and y are positive.
    """
    x, y = np.asarray(x), np.asarray(y)  # Ensure inputs are arrays
    x, y = np.maximum(x, 1e-10), np.maximum(y, 1e-10)  # Avoid zero/negative values

    if np.isnan(alpha) or np.isinf(alpha):
        raise ValueError("Invalid alpha value: Must be a finite number.")
    
    return k * (x**alpha + y**alpha)

def gauss_quadrature(N):
    """
    Computes Gauss-Legendre quadrature nodes and weights.
    """
    nodes, weights = leggauss(N)
    return nodes, weights

def moment_evolution(t, M, k, alpha, N):
    """
    Computes dM/dt for given moments using QMOM.
    """
    dMdt = np.zeros_like(M)

    # Compute quadrature nodes & weights
    nodes, weights = gauss_quadrature(N)
    
    for n in range(len(M)):
        sum_term = 0.0
        for i in range(n + 1):
            # Ensure indices are within valid range
            if i < len(nodes) and (n - i) < len(nodes):
                beta_ij = aggregation_kernel(nodes[i], nodes[n - i], k, alpha)
                sum_term += (M[i] * M[n - i] * beta_ij)
        
        dMdt[n] = 0.5 * sum_term

    return dMdt

def qmom_solver(t_span, M0, k, alpha, N=4):
    """
    Solves the QMOM system for given initial moments.
    """
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    
    sol = spi.solve_ivp(moment_evolution, t_span, M0, args=(k, alpha, N), 
                         method='RK45', t_eval=t_eval)

    return sol.t, sol.y.T  # Return time points & moment solutions

def generate_dataset(num_samples=10000, t_span=(0, 10), N=4):
    """
    Generates a dataset for PINN/XGBoost training by solving QMOM 
    for multiple random initial conditions and kernel parameters.
    
    Returns:
    - A Pandas DataFrame
    - Saves the dataset as 'qmom_dataset.csv'
    """
    data = []

    for _ in range(num_samples):
        # Randomly initialize moments (positive values)
        M0 = np.abs(np.random.uniform(0.1, 2.0, size=N))
        
        # Randomly select kernel parameters
        k = np.random.uniform(0.5, 2.0)
        alpha = np.random.uniform(0.5, 2.0)

        # Solve QMOM
        t, M_sol = qmom_solver(t_span, M0, k, alpha, N)

        for i, t_val in enumerate(t):
            data.append([t_val, *M0, k, alpha, *M_sol[i]])

    # Convert to DataFrame
    columns = ["time"] + [f"M{i}_initial" for i in range(N)] + ["k", "alpha"] + [f"M{i}_t" for i in range(N)]
    df = pd.DataFrame(data, columns=columns)

    # Save data to CSV
    df.to_csv("qmom_dataset.csv", index=False)
    print("Dataset saved as 'qmom_dataset.csv'")

    return df

if __name__ == "__main__":
    #initial conditions
    M0 = [1.0, 0.5, 0.2, 0.1]  # Initial moments
    t_span = (0, 10)  # Simulate from t=0 to t=10
    k, alpha = 1.0, 1.0  # Kernel parameters

    #Run QMOM solver
    t, M_sol = qmom_solver(t_span, M0, k, alpha, N=4)

    # Printing final moment values
    print("Final Moments:", " ".join(f"{m:.6f}" for m in M_sol[-1]))

    # Generating dataset
    generate_dataset(num_samples=1000)  # You can change the sample size
