import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
file_path = "Medical_insurance.csv"
df = pd.read_csv(file_path)

# Encode categorical variables (hardcoded mapping txt->int)
df['sex'] = df['sex'].map({'male': 1, 'female': 2})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df['region'] = df['region'].map({'southeast': 1, 'northeast': 2, 'southwest': 3, 'northwest': 4})

# A. Feature Normalization (exclude last column 'charges')
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  # Normalize all features except the target (last col)

# Separate features and target
X_multi = df.drop(columns=['charges']).values  # Features (all except 'charges')
X_single = df[['bmi']].values  # Single feature model
y = df['charges'].values  # Target variable (DO NOT normalize)

# Train-test split (80% train, 20% test)
X_train_multi, X_test_multi, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)
X_train_single, X_test_single = X_train_multi[:, 1:2], X_test_multi[:, 1:2]  # Using only 'bmi'

# B. Train Linear Regression (Single)
lin_reg_single = LinearRegression()
lin_reg_single.fit(X_train_single, y_train)
y_pred_lin_single = lin_reg_single.predict(X_test_single)
mae_lin_single = mean_absolute_error(y_test, y_pred_lin_single)

# C. Train Linear Regression (Multiple)
lin_reg_multi = LinearRegression()
lin_reg_multi.fit(X_train_multi, y_train)
y_pred_lin_multi = lin_reg_multi.predict(X_test_multi)
mae_lin_multi = mean_absolute_error(y_test, y_pred_lin_multi)

# Build a DNN model 
def dnn_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)  # Linear activation
    ])
    model.compile(optimizer='adam', loss='mae')  # Using MAE for regression
    return model

# D. Train DNN (Single) and track training loss
dnn_single = dnn_model((1,))
history_dnn_single = dnn_single.fit(X_train_single, y_train, epochs=100, verbose=0, batch_size=32, validation_split=0.2)
y_pred_dnn_single = dnn_single.predict(X_test_single).flatten()
mae_dnn_single = mean_absolute_error(y_test, y_pred_dnn_single)

# E. Train DNN (Multiple) and track training loss
dnn_multi = dnn_model((X_train_multi.shape[1],))
history_dnn_multi = dnn_multi.fit(X_train_multi, y_train, epochs=100, verbose=0, batch_size=32, validation_split=0.2)
y_pred_dnn_multi = dnn_multi.predict(X_test_multi).flatten()
mae_dnn_multi = mean_absolute_error(y_test, y_pred_dnn_multi)

# F. Compare Performance and Track the Best Model
mae_scores = {
    "Linear Regression (Single Feature)": mae_lin_single,
    "Multiple Linear Regression": mae_lin_multi,
    "DNN (Single Feature)": mae_dnn_single,
    "DNN (Multiple Features)": mae_dnn_multi
}

# Find the best model
best_model_name = min(mae_scores, key=mae_scores.get) # Best = lowest MAE
best_mae = mae_scores[best_model_name]
print("\n=== Model Performance Comparison ===")
for model, mae in mae_scores.items():
    print(f"{model}: MAE = {mae:.2f}")
print(f"\nBest Model: {best_model_name} with MAE = {best_mae:.2f}")

# G. Visualization of predictions from the best-performing model
best_predictions = {
    "Linear Regression (Single Feature)": y_pred_lin_single,
    "Multiple Linear Regression": y_pred_lin_multi,
    "DNN (Single Feature)": y_pred_dnn_single,
    "DNN (Multiple Features)": y_pred_dnn_multi
}[best_model_name]

# Assume linear models have low MAE anyway
best_loss = {
    "DNN (Single Feature)": history_dnn_single,
    "DNN (Multiple Features)": history_dnn_multi
}[best_model_name]

# Visualise the best model from all
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions, color='blue', alpha=0.5, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed', label="Ideal Fit")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title(f"Best Model: {best_model_name} - Predictions vs Actual Values")
plt.legend()
plt.show()

# Plot Loss per Epoch (curve)
plt.figure(figsize=(12, 5))

# Loss curve for best model
plt.subplot(1, 2, 2)
plt.plot(best_loss.history['loss'], label='Training Loss')
plt.plot(best_loss.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title(f"Training Loss - {best_model_name}")
plt.legend()

plt.show()
