# ml_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
df = pd.read_csv('tripadvisor_review.csv')

# Create target variable (average of all categories)
category_cols = [col for col in df.columns if 'Category' in col]
df['Average_Score'] = df[category_cols].mean(axis=1)

# Features and target
X = df[category_cols]
y = df['Average_Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.3f}')
print(f'R2 Score: {r2:.3f}')

# Save the model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
