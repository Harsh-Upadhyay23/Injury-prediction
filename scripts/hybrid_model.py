import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def create_voting_classifier():
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft'  # Use probability predictions for voting
    )
    return voting_clf

def create_stacking_classifier():
    # Base estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ]
    
    # Final estimator
    final_estimator = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5  # 5-fold cross-validation
    )
    return stacking_clf

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    
    print(f"\nModel: {model_name}")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Classification Report:\n", report)
    
    return model, acc, f1

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results/hybrid_models", exist_ok=True)
    
    # Load preprocessed and balanced data
    print("Loading data...")
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
    
    # Create and train models for Severity
    print("\n===== Training Hybrid Models for Injury Severity =====")
    
    # Voting Classifier
    voting_clf_severity = create_voting_classifier()
    voting_model_s, voting_acc_s, voting_f1_s = train_and_evaluate(
        voting_clf_severity, X_train_s, X_test_s, y_train_s, y_test_s, "Voting Classifier (Severity)"
    )
    
    # Stacking Classifier
    stacking_clf_severity = create_stacking_classifier()
    stacking_model_s, stacking_acc_s, stacking_f1_s = train_and_evaluate(
        stacking_clf_severity, X_train_s, X_test_s, y_train_s, y_test_s, "Stacking Classifier (Severity)"
    )
    
    # Create and train models for Location
    print("\n===== Training Hybrid Models for Injury Location =====")
    
    # Voting Classifier
    voting_clf_location = create_voting_classifier()
    voting_model_l, voting_acc_l, voting_f1_l = train_and_evaluate(
        voting_clf_location, X_train_l, X_test_l, y_train_l, y_test_l, "Voting Classifier (Location)"
    )
    
    # Stacking Classifier
    stacking_clf_location = create_stacking_classifier()
    stacking_model_l, stacking_acc_l, stacking_f1_l = train_and_evaluate(
        stacking_clf_location, X_train_l, X_test_l, y_train_l, y_test_l, "Stacking Classifier (Location)"
    )
    
    # Save models
    print("\nSaving models...")
    joblib.dump(voting_model_s, "results/hybrid_models/voting_severity_model.pkl")
    joblib.dump(stacking_model_s, "results/hybrid_models/stacking_severity_model.pkl")
    joblib.dump(voting_model_l, "results/hybrid_models/voting_location_model.pkl")
    joblib.dump(stacking_model_l, "results/hybrid_models/stacking_location_model.pkl")
    
    # Create and save leaderboard
    results = {
        'Model': ['Voting (Severity)', 'Stacking (Severity)', 'Voting (Location)', 'Stacking (Location)'],
        'Accuracy': [voting_acc_s, stacking_acc_s, voting_acc_l, stacking_acc_l],
        'F1 Score': [voting_f1_s, stacking_f1_s, voting_f1_l, stacking_f1_l]
    }
    
    results_df = pd.DataFrame(results)
    print("\n=== Hybrid Models Leaderboard ===")
    print(results_df)
    results_df.to_csv("results/hybrid_models/hybrid_models_leaderboard.csv", index=False)

if __name__ == "__main__":
    main() 