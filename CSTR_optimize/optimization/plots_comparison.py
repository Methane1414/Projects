import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
k0 = 1e3     # Example value for pre-exponential factor
E = 8000       # Example value for activation energy

# Importing synthetic dataset
data = pd.read_csv("C:\\Users\\chaur\\Documents\\Projects\\CSTR_optimize\\cpp_files\\Sythetic_dataset.csv")

# Extracting input (T, tau, F_A0) and target (X)
X_data = data[['T', 'tau', 'F_A0']].values
y_data = data['X'].values

# Train-test split for reliable evaluation
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Defining the polynomial degrees to test
degrees = [2, 3, 4, 5]

#To Store results for comparison
results = {}
scalers = {}
models = {}
polynomial_features = {}
test_errors = []

for degree in degrees:
    # Generating polynomial features
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Feature scaling (Standardization)
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly = scaler.transform(X_test_poly)

    # Saving scaler and poly for future use
    scalers[degree] = scaler
    polynomial_features[degree] = poly

    # Model (Using Ridge Regression to prevent overfitting)
    model = Ridge(alpha=0.1)
    model.fit(X_train_poly, y_train)
    models[degree] = model

    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Performance Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    results[degree] = {
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train R²': r2_train,
        'Test R²': r2_test
    }
    test_errors.append(test_mse)

    # Print results
    print(f"\nDegree {degree}:")
    print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
    print(f"Train R²: {r2_train:.6f}, Test R²: {r2_test:.6f}")

    # Plot True vs. Predicted for Test Data
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('True Conversion (X_true)')
    plt.ylabel('Predicted Conversion (X_pred)')
    plt.title(f'Polynomial Regression (Degree {degree}) - True vs. Predicted (Test Data)')
    plt.grid(True)
    plt.show()

# Saving results to CSV file
results_df = pd.DataFrame.from_dict(results, orient='index')
output_file = 'polynomial_regression_comparison.csv'
results_df.to_csv(output_file, index=True)
print(f"\nPerformance report saved to '{output_file}'")

# ===========================
# Error vs. Degree Plot
# ===========================

plt.figure(figsize=(10, 6))
plt.plot(degrees, test_errors, marker='o', linestyle='-', color='blue')
plt.title('Test MSE vs. Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Test Mean Squared Error')
plt.grid(True)
plt.show()

# ===========================
# Newton's Method Implementation (Comparison)
# ===========================

def newtons_method(T, tau, F_A0):
    R = 8.314  # Universal gas constant, J/mol.K
    k = k0 * np.exp(-E / (R * T))
    X = 1 - np.exp(-k * tau / F_A0)
    return X

# Apply Newton's Method to Test Set
y_test_newton = np.array([newtons_method(T, tau, F_A0) for T, tau, F_A0 in X_test])

# Performance Metrics for Newton's Method
newton_test_mse = mean_squared_error(y_test, y_test_newton)
newton_r2_test = r2_score(y_test, y_test_newton)

print("\nNewton's Method Results (on Test Data):")
print(f"Test MSE: {newton_test_mse:.6f}")
print(f"Test R²: {newton_r2_test:.6f}")

# Plotting Newton's Method vs. True Data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_newton, alpha=0.5, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('True Conversion (X_true)')
plt.ylabel('Predicted Conversion (X_pred) by Newton\'s Method')
plt.title("Newton's Method - True vs. Predicted (Test Data)")
plt.grid(True)
plt.show()
