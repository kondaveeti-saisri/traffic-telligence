import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# Load processed data
data = pd.read_csv('processed_traffic_data.csv')

# Features and target
X = data.drop('traffic_volume', axis=1)
y = data['traffic_volume']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define features
numeric_features = ['temp', 'rain', 'snow', 'year', 'month', 'day', 'hour', 'minute', 'second']
categorical_features = ['holiday', 'weather']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Save model
joblib.dump(model, 'model.pkl')

# Save preprocessor separately if needed
preprocessor = model.named_steps['preprocessor']
joblib.dump(preprocessor, 'preprocessor.pkl')

print("✅ Model saved as model.pkl")
print("✅ Preprocessor saved as preprocessor.pkl")