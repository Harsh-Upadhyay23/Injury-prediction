import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def create_ensemble_1():
    # Combination 1: RF + XGB + SVM
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    svm = SVC(probability=True, kernel='rbf', C=1.0)
    
    return VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('svm', svm)],
        voting='soft',
        weights=[1, 1, 0.5]  # Give SVM less weight
    )

def create_ensemble_2():
    # Combination 2: RF + XGB + Neural Network
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    
    return VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('nn', nn)],
        voting='soft',
        weights=[1, 1, 0.8]
    )

def create_ensemble_3():
    # Combination 3: Stacking with RF, XGB, and SVM
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
        ('svm', SVC(probability=True, kernel='rbf', C=1.0))
    ]
    
    final_estimator = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5
    )

def create_ensemble_4():
    # Combination 4: Bagging with XGBoost
    return BaggingClassifier(
        estimator=XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        n_estimators=10,
        random_state=42
    )

def create_ensemble_5():
    # Combination 5: RF + XGB + SVM + Neural Network
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    svm = SVC(probability=True, kernel='rbf', C=1.0)
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    
    return VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('svm', svm), ('nn', nn)],
        voting='soft',
        weights=[1, 1, 0.5, 0.8]
    )

def create_ensemble_6():
    """
    NEW: Adaptive Boosting + Gradient Boosting + Extra Trees
    Uses different boosting and tree-based ensemble methods
    """
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    return VotingClassifier(
        estimators=[('ada', ada), ('gb', gb), ('et', et)],
        voting='soft',
        weights=[1.2, 1.0, 0.8]
    )

def create_ensemble_7():
    """
    NEW: Blending with Meta-Learning
    Uses a two-level approach with neural network as meta-learner
    """
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
        ('svm', SVC(probability=True, kernel='rbf', C=1.0)),
        ('nn', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ]
    
    meta_learner = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=1000,
        random_state=42
    )
    
    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba'
    )

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
    os.makedirs("results/updated_advanced_hybrid", exist_ok=True)
    
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
    
    # Define all ensembles (including new ones)
    ensembles = {
        "Ensemble 1 (RF+XGB+SVM)": create_ensemble_1(),
        "Ensemble 2 (RF+XGB+NN)": create_ensemble_2(),
        "Ensemble 3 (Stacking)": create_ensemble_3(),
        "Ensemble 4 (Bagging)": create_ensemble_4(),
        "Ensemble 5 (RF+XGB+SVM+NN)": create_ensemble_5(),
        "Ensemble 6 (AdaBoost+GB+ExtraTrees)": create_ensemble_6(),
        "Ensemble 7 (Blending+MetaLearning)": create_ensemble_7()
    }
    
    # Train and evaluate for Severity
    print("\n===== Training All Ensembles for Injury Severity =====")
    results_severity = []
    for name, model in ensembles.items():
        model, acc, f1 = train_and_evaluate(model, X_train_s, X_test_s, y_train_s, y_test_s, name)
        results_severity.append((name, acc, f1))
        joblib.dump(model, f"results/updated_advanced_hybrid/{name.replace(' ', '_').lower()}_severity_model.pkl")
    
    # Train and evaluate for Location
    print("\n===== Training All Ensembles for Injury Location =====")
    results_location = []
    for name, model in ensembles.items():
        model, acc, f1 = train_and_evaluate(model, X_train_l, X_test_l, y_train_l, y_test_l, name)
        results_location.append((name, acc, f1))
        joblib.dump(model, f"results/updated_advanced_hybrid/{name.replace(' ', '_').lower()}_location_model.pkl")
    
    # Create and save leaderboards
    results_df_s = pd.DataFrame(results_severity, columns=["Model", "Accuracy", "F1 Score"])
    results_df_l = pd.DataFrame(results_location, columns=["Model", "Accuracy", "F1 Score"])
    
    results_df_s = results_df_s.sort_values(by="F1 Score", ascending=False)
    results_df_l = results_df_l.sort_values(by="F1 Score", ascending=False)
    
    print("\n--- Updated Advanced Ensembles Severity Leaderboard ---")
    print(results_df_s)
    print("\n--- Updated Advanced Ensembles Location Leaderboard ---")
    print(results_df_l)
    
    # Save leaderboards
    results_df_s.to_csv("results/updated_advanced_hybrid/updated_ensembles_severity_leaderboard.csv", index=False)
    results_df_l.to_csv("results/updated_advanced_hybrid/updated_ensembles_location_leaderboard.csv", index=False)
    
    print("\nAll ensemble models training completed!")
    print("Models saved in results/updated_advanced_hybrid/")

if __name__ == "__main__":
    main()
