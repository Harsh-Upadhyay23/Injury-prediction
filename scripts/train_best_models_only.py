#!/usr/bin/env python3
"""
Train only the best performing models for injury prediction
Based on comprehensive analysis, the best models are:
- Severity: Ensemble 2 (RF+XGB+NN) - F1: 0.9275
- Location: Random Forest - F1: 0.7842
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def create_best_severity_model():
    """Create the best performing severity model: Ensemble 2 (RF+XGB+NN)"""
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    
    return VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('nn', nn)],
        voting='soft',
        weights=[1, 1, 0.8]
    )

def create_best_location_model():
    """Create the best performing location model: Random Forest"""
    return RandomForestClassifier(n_estimators=200, random_state=42)

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, task_type):
    """Train and evaluate a model"""
    print(f"\n🚀 Training {model_name} for {task_type}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    
    print(f"✅ {model_name} ({task_type}) Results:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    return model, acc, f1

def main():
    """Main function to train only the best models"""
    print("🎯 Training BEST Models Only for Injury Prediction")
    print("=" * 60)
    print("Severity: Ensemble 2 (RF+XGB+NN) - Expected F1: 0.9275")
    print("Location: Random Forest - Expected F1: 0.7842")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results/best_models", exist_ok=True)
    
    # Load preprocessed and balanced data
    print("\n📊 Loading data...")
    df = pd.read_excel("data/balanced_sheet1.xlsx")
    
    # Separate features and targets
    X = df.drop(columns=["Injury Severity", "Injury Location"])
    y_severity = df["Injury Severity"]
    y_location = df["Injury Location"]
    
    # Encode Injury Location
    location_encoder = LabelEncoder()
    y_location_encoded = location_encoder.fit_transform(y_location)
    y_location = y_location_encoded
    
    # Split datasets
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_severity, test_size=0.2, random_state=42)
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y_location, test_size=0.2, random_state=42)
    
    print(f"📈 Data split complete:")
    print(f"   Training samples: {X_train_s.shape[0]}")
    print(f"   Test samples: {X_test_s.shape[0]}")
    print(f"   Features: {X_train_s.shape[1]}")
    
    # Train best severity model
    print("\n" + "="*60)
    print("🏆 TRAINING BEST SEVERITY MODEL")
    print("="*60)
    
    severity_model = create_best_severity_model()
    severity_model, severity_acc, severity_f1 = train_and_evaluate(
        severity_model, X_train_s, X_test_s, y_train_s, y_test_s, 
        "Ensemble 2 (RF+XGB+NN)", "Severity"
    )
    
    # Train best location model
    print("\n" + "="*60)
    print("🏆 TRAINING BEST LOCATION MODEL")
    print("="*60)
    
    location_model = create_best_location_model()
    location_model, location_acc, location_f1 = train_and_evaluate(
        location_model, X_train_l, X_test_l, y_train_l, y_test_l, 
        "Random Forest", "Location"
    )
    
    # Save models
    print("\n💾 Saving best models...")
    joblib.dump(severity_model, "results/best_models/best_severity_model.pkl")
    joblib.dump(location_model, "results/best_models/best_location_model.pkl")
    joblib.dump(location_encoder, "results/best_models/location_encoder.pkl")
    
    # Save results summary
    results_summary = {
        'severity_model': {
            'name': 'Ensemble 2 (RF+XGB+NN)',
            'accuracy': severity_acc,
            'f1_score': severity_f1,
            'model_path': 'results/best_models/best_severity_model.pkl'
        },
        'location_model': {
            'name': 'Random Forest',
            'accuracy': location_acc,
            'f1_score': location_f1,
            'model_path': 'results/best_models/best_location_model.pkl'
        }
    }
    
    import json
    with open('results/best_models/best_models_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 BEST MODELS TRAINING COMPLETE")
    print("="*60)
    print(f"✅ Severity Model: {results_summary['severity_model']['name']}")
    print(f"   F1-Score: {results_summary['severity_model']['f1_score']:.4f}")
    print(f"   Accuracy: {results_summary['severity_model']['accuracy']:.4f}")
    print(f"   Saved to: {results_summary['severity_model']['model_path']}")
    
    print(f"\n✅ Location Model: {results_summary['location_model']['name']}")
    print(f"   F1-Score: {results_summary['location_model']['f1_score']:.4f}")
    print(f"   Accuracy: {results_summary['location_model']['accuracy']:.4f}")
    print(f"   Saved to: {results_summary['location_model']['model_path']}")
    
    print(f"\n📁 All models saved in: results/best_models/")
    print("🚀 Ready for production use!")

if __name__ == "__main__":
    main()
