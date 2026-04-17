import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def train_ml_baseline():
    print("--- Building Classical ML Baseline (XGBoost) ---")
    
    # 1. Simulate Data (Replace this block with pd.read_csv('your_data.csv'))
    print("1. Loading Data...")
    num_samples = 1000
    
    # Fake genomic features (e.g., 50 important SNPs)
    genomics = np.random.rand(num_samples, 50) 
    
    # Fake environmental features (e.g., Temperature, Precipitation)
    environment = np.random.rand(num_samples, 5) 
    
    # Combine features into one dataset X
    X = np.hstack((genomics, environment))
    
    # Fake target variable (e.g., Crop Yield)
    # Yield is somewhat dependent on the first few genes and the temperature
    y = 50 + (X[:, 0] * 20) - (X[:, 50] * 10) + np.random.randn(num_samples) * 5
    
    # 2. Preprocess Data
    print("2. Preprocessing Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train the XGBoost Model
    print("3. Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 4. Evaluate the Model
    print("4. Evaluating Model...")
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"-> Mean Squared Error (MSE): {mse:.2f}")
    print(f"-> R-squared (Accuracy): {r2:.2f}")
    
    # 5. Feature Importance (Interpretability)
    print("\n5. Feature Importance (Top 3):")
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-3:][::-1]
    for i, idx in enumerate(top_indices):
        feature_name = f"Environment Variable {idx-50}" if idx >= 50 else f"Gene Marker {idx}"
        print(f"   #{i+1}: {feature_name} (Score: {importances[idx]:.3f})")

if __name__ == "__main__":
    train_ml_baseline()
