import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('C:\\Users\\chaur\\Documents\\Projects\\QMOM_PINN\\qmom_dataset.csv')

# Define feature and target columns
feature_cols = ['time', 'k', 'alpha'] + [f'M{i}_initial' for i in range(4)]
target_cols = [f'M{i}_t' for i in range(4)]

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

# Define a simple neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input layer
    Dense(64, activation='tanh'),
    Dense(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(Y_train.shape[1], activation='linear')  # Output layer
])

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X_train, Y_train, epochs=500, batch_size=32, validation_data=(X_test, Y_test))

# Save model
model.save("qmom_pinn_simple.h5")

# Predict and print a few values
Y_pred_test = model.predict(X_test)
predictions_original = scaler_Y.inverse_transform(Y_pred_test[:5])
print("Model trained! Sample predictions:\n", predictions_original)
