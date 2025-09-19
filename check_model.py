import joblib
import pandas as pd

# Load the model
model = joblib.load('results/advanced_hybrid/ensemble_2_(rf+xgb+nn)_severity_model.pkl')

# Print model's feature names
if hasattr(model, 'feature_names_in_'):
    print("Model's feature names (one per line):")
    for name in model.feature_names_in_:
        print(f"- {name}")
else:
    print("Model doesn't have feature_names_in_ attribute")

# Load a sample of the training data to see feature names
df = pd.read_excel("data/balanced_sheet1.xlsx")
print("\nFeatures from training data (one per line):")
for col in df.columns:
    print(f"- {col}") 