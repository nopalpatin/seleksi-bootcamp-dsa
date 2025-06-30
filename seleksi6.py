from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load data
df = pd.read_csv("train.csv")

# Feature engineering: date parsing
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df = df.drop(columns=['ID', 'date'])

# One-hot encoding for cluster_id
df = pd.get_dummies(df, columns=['cluster_id'])

# Pisahkan fitur dan target
X = df.drop(columns=['electricity_consumption'])
y = df['electricity_consumption']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluasi hasil
best_model = grid_search.best_estimator_
rmse_train = mean_squared_error(y_train, best_model.predict(X_train), squared=False)
rmse_test = mean_squared_error(y_test, best_model.predict(X_test), squared=False)

print("Best Parameters:", grid_search.best_params_)
print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)
