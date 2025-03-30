import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants (You should define these based on your system)
k0 = 1e3       # Example value for pre-exponential factor
E = 8000       # Example value for activation energy

# Load your synthetic dataset
data = pd.read_csv("C:\\Users\\chaur\\Documents\\Projects\\CSTR_optimize\\cpp_files\\Sythetic_dataset.csv")

# Extract input features (T, tau, F_A0) and target (X)
X_data = data[['T', 'tau', 'F_A0']].values
y_data = data['X'].values

# Train-test split for reliable evaluation
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Define the polynomial degrees to test
degrees = [2, 3, 4, 5]

# Store results for comparison
results = {}
scalers = {}
models = {}
polynomial_features = {}

for degree in degrees:
    # Generate polynomial features
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Feature scaling (Standardization)
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly = scaler.transform(X_test_poly)

    # Save scaler and poly for future use
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

    # Print results
    print(f"\nDegree {degree}:")
    print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
    print(f"Train R²: {r2_train:.6f}, Test R²: {r2_test:.6f}")

# Save results to CSV file
results_df = pd.DataFrame.from_dict(results, orient='index')
output_file = 'polynomial_regression_comparison.csv'
results_df.to_csv(output_file, index=True)
print(f"\nPerformance report saved to '{output_file}'")

# ===========================
# User Input Prediction
# ===========================

def predict_with_model(T, tau, F_A0, degree):
    if degree not in models:
        print(f"Model of degree {degree} is not trained. Please choose a degree from {list(models.keys())}.")
        return None

    model = models[degree]
    poly = polynomial_features[degree]
    scaler = scalers[degree]

    # Prepare input data
    user_input = np.array([[T, tau, F_A0]])
    user_input_poly = poly.transform(user_input)
    user_input_scaled = scaler.transform(user_input_poly)

    # Make prediction
    prediction = model.predict(user_input_scaled)
    return prediction[0]

# ===========================
# Allow User to Test Model
# ===========================

while True:
    try:
        print("\nTest the trained model with your own data:")
        T = float(input("Enter temperature (T): "))
        tau = float(input("Enter residence time (tau): "))
        F_A0 = float(input("Enter initial molar flow rate (F_A0): "))
        degree = int(input(f"Enter polynomial degree to use (Choose from {degrees}): "))

        prediction = predict_with_model(T, tau, F_A0, degree)
        if prediction is not None:
            print(f"\nPredicted Conversion (X) for T={T}, tau={tau}, F_A0={F_A0}, Degree={degree}: {prediction:.6f}")

        another = input("\nDo you want to test another input? (yes/no): ").strip().lower()
        if another != 'yes':
            break
    except ValueError:
        print("Invalid input. Please enter numeric values for T, tau, F_A0, and an available degree.")
