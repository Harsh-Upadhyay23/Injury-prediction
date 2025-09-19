import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load preprocessed and balanced data
    print("Loading data...")
    df = pd.read_excel("data/balanced_sheet1.xlsx")
    
    # Save feature names
    feature_names = list(df.drop(columns=["Injury Severity", "Injury Location"]).columns)
    joblib.dump(feature_names, "results/feature_names.pkl")
    
    # Separate features and targets
    X = df.drop(columns=["Injury Severity", "Injury Location"])
    y_severity = df["Injury Severity"]
    y_location = df["Injury Location"]
    
    # Encode Injury Location
    location_encoder = LabelEncoder()
    y_location_encoded = location_encoder.fit_transform(y_location)
    joblib.dump(location_encoder, "results/location_encoder.pkl")
    
    # Split datasets
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_severity, test_size=0.2, random_state=42)
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y_location, test_size=0.2, random_state=42)
    
    # Train severity model
    print("Training severity model...")
    severity_model = RandomForestClassifier(n_estimators=200, random_state=42)
    severity_model.fit(X_train_s, y_train_s)
    joblib.dump(severity_model, "results/severity_model.pkl")
    
    # Train location model
    print("Training location model...")
    location_model = RandomForestClassifier(n_estimators=200, random_state=42)
    location_model.fit(X_train_l, y_train_l)
    joblib.dump(location_model, "results/location_model.pkl")
    
    print("Training complete. Models and feature names saved.")

if __name__ == "__main__":
    main() 