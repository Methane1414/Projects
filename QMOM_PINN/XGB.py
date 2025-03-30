import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('C:\\Users\\chaur\\Documents\\Projects\\QMOM_PINN\\qmom_dataset.csv')

# Define feature and target columns
feature_cols = ["time", "k", "alpha"] + [f"M{i}_initial" for i in range(4)]
target_cols = [f"M{i}_t" for i in range(4)]

# Prepare input (X) and output (Y)
X = df[feature_cols].values
Y = df[target_cols].values

# Normalize data
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Train XGBoost models for each target
models = {}
for i, target in enumerate(target_cols):
    print(f"Training XGBoost for {target}...")
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.1, max_depth=5)
    model.fit(X_train, Y_train[:, i])
    models[target] = model
    joblib.dump(model, f"xgboost_{target}.pkl")

print("All models trained and saved!")

# Load models and make predictions
loaded_models = {target: joblib.load(f"xgboost_{target}.pkl") for target in target_cols}
Y_pred_test = np.column_stack([loaded_models[target].predict(X_test) for target in target_cols])

# Convert predictions back to original scale
Y_pred_test_original = scaler_Y.inverse_transform(Y_pred_test)

# Compute RMSE
rmse_test = np.sqrt(mean_squared_error(scaler_Y.inverse_transform(Y_test), Y_pred_test_original))

print(f"Testing RMSE: {rmse_test:.4f}")
print("Sample Predictions:\n", Y_pred_test_original[:5])
