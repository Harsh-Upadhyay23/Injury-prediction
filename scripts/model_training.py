import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import joblib
import os
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
import re
warnings.filterwarnings('ignore')


def create_hybrid_model():
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    hybrid_clf = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft'
    )
    return hybrid_clf


def create_hybrid_model_2():
    """Hybrid Model 2: AdaBoost + Gradient Boosting + Extra Trees"""
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    
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


def create_hybrid_model_3():
    """Hybrid Model 3: Blending with Meta-Learning"""
    from sklearn.ensemble import StackingClassifier
    
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')),
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


def create_lgbm_hybrid_model():
    """Hybrid Model 4: LightGBM + XGBoost + Random Forest"""
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=6,
        random_state=42,
        verbosity=-1,
        force_col_wise=True  # This helps with compatibility issues
    )
    
    return VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft',
        weights=[1.0, 1.2, 1.1]
    )


# Load preprocessed and balanced data
df = pd.read_excel("data/balanced_sheet1.xlsx")

# Separate features and targets
X = df.drop(columns=["Injury Severity", "Injury Location"])
y_severity = df["Injury Severity"]
y_location = df["Injury Location"]

# Encode Injury Location and save encoder
location_encoder = LabelEncoder()
y_location_encoded = location_encoder.fit_transform(y_location)
joblib.dump(location_encoder, "results/models/location_encoder.pkl")

# Use encoded y_location
y_location = y_location_encoded

# Ensure all values are non-negative for compatibility
if y_location.min() < 0:
    shift = abs(y_location.min())
    y_location += shift
    print(f"Shifted Injury Location values up by {shift} to make all classes non-negative.")

# Split dataset for both targets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_severity, test_size=0.2, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y_location, test_size=0.2, random_state=42)

# Check data types and convert if necessary
print("Data info:")
print(f"X_train_s shape: {X_train_s.shape}")
print(f"X_train_s dtypes: {X_train_s.dtypes.unique()}")
print(f"y_train_s unique values: {len(np.unique(y_train_s))}")
print(f"y_train_l unique values: {len(np.unique(y_train_l))}")

# Ensure numeric data for LightGBM
X_train_s = X_train_s.astype(float)
X_test_s = X_test_s.astype(float)
X_train_l = X_train_l.astype(float)
X_test_l = X_test_l.astype(float)

# Sanitize feature names for LightGBM compatibility
def _sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    # Replace any non-alphanumeric/underscore characters with underscore
    sanitized_columns = []
    seen = {}
    for col in df.columns:
        new_col = re.sub(r"[^0-9A-Za-z_]", "_", str(col))
        # Avoid empty names
        if new_col == "":
            new_col = "feature"
        # Ensure uniqueness
        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0
        sanitized_columns.append(new_col)
    df_sanitized = df.copy()
    df_sanitized.columns = sanitized_columns
    return df_sanitized

# Precompute sanitized versions (to be used for LightGBM-containing models)
X_train_s_sanitized = _sanitize_feature_names(X_train_s)
X_test_s_sanitized = _sanitize_feature_names(X_test_s)
X_train_l_sanitized = _sanitize_feature_names(X_train_l)
X_test_l_sanitized = _sanitize_feature_names(X_test_l)

# Define models with proper LightGBM configuration
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100,  # Reduced for faster training
        learning_rate=0.1,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        force_col_wise=True,  # Important for compatibility
        objective='multiclass'  # Will be set dynamically
    ),
    "Hybrid (RF + XGB)": create_hybrid_model(),
    "Hybrid 2 (AdaBoost+GB+ExtraTrees)": create_hybrid_model_2(),
    "Hybrid 3 (Blending+MetaLearning)": create_hybrid_model_3(),
    "Hybrid 4 (RF+XGB+LGBM)": create_lgbm_hybrid_model()
}

os.makedirs("results/models", exist_ok=True)

# Train and evaluate for Injury Severity
print("\n===== Training for Injury Severity =====")
results_severity = []

for name, model in models.items():
    print(f"\nTraining {name} for Severity...")
    try:
        # Special handling for LightGBM to set objective dynamically
        if name == "LightGBM":
            n_classes = len(np.unique(y_train_s))
            if n_classes == 2:
                model.set_params(objective='binary')
            else:
                model.set_params(objective='multiclass', num_class=n_classes)
        
        # Fit the model
        if name in ["LightGBM", "Hybrid 4 (RF+XGB+LGBM)"]:
            model.fit(X_train_s_sanitized, y_train_s)
            y_pred = model.predict(X_test_s_sanitized)
        else:
            model.fit(X_train_s, y_train_s)
            y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test_s, y_pred)
        f1 = f1_score(y_test_s, y_pred, average='weighted')
        report = classification_report(y_test_s, y_pred)

        print(f"Model: {name} (Severity)")
        print(f"Accuracy: {acc:.6f}")
        print(f"F1 Score: {f1:.6f}")

        results_severity.append((name, acc, f1))
        
        # Save model with cleaned filename
        filename = f"results/models/{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_').lower()}_severity_model.pkl"
        joblib.dump(model, filename)
        print(f"Saved model to {filename}")
    
    except Exception as e:
        print(f"ERROR training {name} for severity: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        continue

# Train and evaluate for Injury Location  
print("\n===== Training for Injury Location =====")
results_location = []

for name, model in models.items():
    print(f"\nTraining {name} for Location...")
    try:
        # Special handling for LightGBM to set objective dynamically
        if name == "LightGBM":
            n_classes = len(np.unique(y_train_l))
            if n_classes == 2:
                model.set_params(objective='binary')
            else:
                model.set_params(objective='multiclass', num_class=n_classes)
        
        # Fit the model
        if name in ["LightGBM", "Hybrid 4 (RF+XGB+LGBM)"]:
            model.fit(X_train_l_sanitized, y_train_l)
            y_pred = model.predict(X_test_l_sanitized)
        else:
            model.fit(X_train_l, y_train_l)
            y_pred = model.predict(X_test_l)

        acc = accuracy_score(y_test_l, y_pred)
        f1 = f1_score(y_test_l, y_pred, average='weighted')
        report = classification_report(y_test_l, y_pred)

        print(f"Model: {name} (Location)")
        print(f"Accuracy: {acc:.6f}")
        print(f"F1 Score: {f1:.6f}")

        results_location.append((name, acc, f1))
        
        # Save model with cleaned filename
        filename = f"results/models/{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_').lower()}_location_model.pkl"
        joblib.dump(model, filename)
        print(f"Saved model to {filename}")
    
    except Exception as e:
        print(f"ERROR training {name} for location: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        continue

# Create and display leaderboards
results_df_s = pd.DataFrame(results_severity, columns=["Model", "Accuracy", "F1 Score"])
results_df_l = pd.DataFrame(results_location, columns=["Model", "Accuracy", "F1 Score"])

results_df_s = results_df_s.sort_values(by="F1 Score", ascending=False)
results_df_l = results_df_l.sort_values(by="F1 Score", ascending=False)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print("\n--- Severity Model Leaderboard ---")
print(results_df_s.to_string(index=False))
print("\n--- Location Model Leaderboard ---")
print(results_df_l.to_string(index=False))

# Save leaderboards
results_df_s.to_csv("results/severity_model_leaderboard.csv", index=False)
results_df_l.to_csv("results/location_model_leaderboard.csv", index=False)

print("\nLeaderboards saved to CSV files.")
