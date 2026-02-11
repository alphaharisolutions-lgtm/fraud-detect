import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_synthetic_data(samples=1000):
    np.random.seed(42)
    # Features: amount, time (hour 0-23), frequency (1-10), location (label encoded 0-4)
    amount = np.random.uniform(10, 20000, samples)
    time = np.random.randint(0, 24, samples)
    frequency = np.random.randint(1, 15, samples)
    location = np.random.randint(0, 5, samples)
    
    # Fraud logic: high amount (>15k) OR late night + high frequency
    fraud_prob = (amount / 20000) * 0.5 + (((time < 5) | (time > 22)) & (frequency > 10)).astype(int) * 0.4
    is_fraud = (fraud_prob + np.random.normal(0, 0.1, samples) > 0.6).astype(int)
    
    data = pd.DataFrame({
        'amount': amount,
        'time': time,
        'frequency': frequency,
        'location': location,
        'is_fraud': is_fraud
    })
    return data

def train_and_save_model():
    print("Generating synthetic transaction data...")
    data = generate_synthetic_data(2000)
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier (Optimized for Python 3.14)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model Training Complete. Test Accuracy: {accuracy:.4f}")
    
    # Save model and scaler
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    joblib.dump(model, 'saved_models/fraud_model.joblib')
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    
    print("Model and Scaler saved successfully in 'saved_models/' directory.")

if __name__ == "__main__":
    train_and_save_model()
