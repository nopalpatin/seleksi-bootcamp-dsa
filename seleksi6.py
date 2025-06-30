import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("train.csv")

# Preprocessing tanggal
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Drop kolom yang tidak digunakan
df = df.drop(columns=['ID', 'date'])

# One-hot encoding untuk cluster
df = pd.get_dummies(df, columns=['cluster_id'])

# Pisahkan fitur dan target
X = df.drop(columns=['electricity_consumption'])
y = df['electricity_consumption']

# Split data train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Model: Random Forest Regressor")
print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)
