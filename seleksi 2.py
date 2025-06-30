import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Load data
df = pd.read_csv("train.csv")

# 2. Pisahkan target
target = 'electricity_consumption'
y = df[target]

# 3. Identifikasi kolom non-numerik
non_numeric_cols = df.select_dtypes(include='object').columns.tolist()
non_numeric_cols.remove('cluster_date') if 'cluster_date' in non_numeric_cols else None

# 4. Pisahkan cluster_date menjadi fitur waktu
if 'cluster_date' in df.columns:
    df['cluster_date'] = pd.to_datetime(df['cluster_date'], errors='coerce')
    df['year'] = df['cluster_date'].dt.year
    df['month'] = df['cluster_date'].dt.month
    df['day'] = df['cluster_date'].dt.day
    df['weekday'] = df['cluster_date'].dt.weekday
    df['dayofyear'] = df['cluster_date'].dt.dayofyear
    df.drop(columns=['cluster_date'], inplace=True)

# 5. Encode kolom kategorikal (jika ada)
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 6. Drop kolom target dari fitur
X = df.drop(columns=[target])

# 7. Bagi data latih dan validasi
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Inisialisasi dan latih model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluasi model
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 10. Plot prediksi vs aktual
plt.figure(figsize=(10, 5))
plt.plot(y_val.values[:100], label='Aktual', marker='o')
plt.plot(y_pred[:100], label='Prediksi', marker='x')
plt.title("Prediksi vs Aktual (Sample 100)")
plt.xlabel("Sample")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
