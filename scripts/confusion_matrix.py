import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load test dataset
test_data = pd.read_csv("data/balanced_sheet1.csv")

# Ensure column names are stripped of spaces
test_data.columns = test_data.columns.str.strip()

# Define features (X) and target (y)
X_test = test_data.drop(columns=['Injury Severity'])  # Drop target column
y_test = test_data['Injury Severity']  # Use correct target column

# Load trained model (replace 'model.pkl' with your actual model file)
import joblib
model = joblib.load("results/xgboost_model.pkl")  # Ensure the correct model file is used

# Make predictions
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
