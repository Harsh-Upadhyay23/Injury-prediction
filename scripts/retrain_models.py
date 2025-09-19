import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib
import os

# Load original and new data
base_data = pd.read_excel("data/balanced_sheet1.xlsx")
new_data_file = "data/new_data_to_add.csv"

if os.path.exists(new_data_file):
    new_data = pd.read_csv(new_data_file)
    print(f"✅ Loaded {len(new_data)} new rows.")
    combined_data = pd.concat([base_data, new_data], ignore_index=True)
else:
    print("⚠️ No new data file found. Using only base data.")
    combined_data = base_data

# Rebuild dataset
X = combined_data.drop(columns=["Injury Severity", "Injury Location"])
y_severity = combined_data["Injury Severity"]
y_location = combined_data["Injury Location"].round().astype(int)

# Fix negative class labels if needed
if y_location.min() < 0:
    shift = abs(y_location.min())
    y_location += shift
    print(f"Shifted Injury Location labels by {shift}.")

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_severity, test_size=0.2, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y_location, test_size=0.2, random_state=42)

# Define and retrain models
print("\n🔄 Retraining Severity Model (Random Forest)...")
severity_model = RandomForestClassifier()
severity_model.fit(X_train_s, y_train_s)
print("Accuracy:", accuracy_score(y_test_s, severity_model.predict(X_test_s)))
print("F1 Score:", f1_score(y_test_s, severity_model.predict(X_test_s), average='weighted'))

print("\n🔄 Retraining Location Model (XGBoost)...")
location_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
location_model.fit(X_train_l, y_train_l)
print("Accuracy:", accuracy_score(y_test_l, location_model.predict(X_test_l)))
print("F1 Score:", f1_score(y_test_l, location_model.predict(X_test_l), average='weighted'))

# Save updated models
joblib.dump(severity_model, "results/models/retrained_severity_model.pkl")
joblib.dump(location_model, "results/models/retrained_location_model.pkl")
print("\n✅ Retrained models saved.")




#Lite model



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Path to your main dataset
DATA_PATH = "data/balanced_sheet1.csv"

# Common reduced features
REDUCED_FEATURES = [
    "Injury Duration (weeks)",
    "Injury Occurred (weeks ago)",
    "Trunk Flexion (cm)",
    "BMI",
    "Weekly Training Hours",
    "Stick Test (cm)",
    "current discomfort / Injury",
    "Shoulder Flexion (deg)",
    "Weight (kg)"
]

# Output model directory
OUTPUT_PATH = "results/models"

def train_lite_model(target_name, output_filename):
    # Load and filter dataset
    df = pd.read_csv(DATA_PATH)

    # Drop any rows with missing values in selected features or target
    df = df.dropna(subset=REDUCED_FEATURES + [target_name])

    X = df[REDUCED_FEATURES]
    y = df[target_name]

    # 🚨 Convert numeric y to string if needed
    if pd.api.types.is_numeric_dtype(y):
        y = y.astype(str)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_PATH, output_filename))

    print(f"✅ Trained and saved {output_filename} using {target_name}")


# Run both models
if __name__ == "__main__":
    train_lite_model("Injury Severity", "lite_model_severity.pkl")
    train_lite_model("Injury Location", "lite_model_location.pkl")
