import joblib
import numpy as np
import pandas as pd

# Load models
print("Loading models...")
severity_model = joblib.load("results/models/retrained_severity_model.pkl")
location_model = joblib.load("results/models/retrained_location_model.pkl")
location_encoder = joblib.load("results/models/location_encoder.pkl")

# Print feature names from the model
print("\nFeature names from severity model:")
if hasattr(severity_model, 'feature_names_in_'):
    print(severity_model.feature_names_in_)
else:
    print("Model does not have feature_names_in_ attribute")

# Create test data with correct feature names
test_data = {
    'Age': 25,
    'Gender': 1,
    'Weight (kg)': 75,
    'Height (m)': 1.75,
    'BMI': 24.5,
    'Waist Circumference (cm)': 85,
    'Hip Circumference (cm)': 95,
    'Quad Circumference (cm)': 55,
    'Calf Circumference (cm)': 38,
    'Upper Arm Circumference (cm)': 32,
    'Wrist Circumference (cm)': 16,
    'Ankle Circumference (cm)': 22,
    'Shoulder Flexion (deg)': 160,
    'Trunk Flexion (cm)': 25,
    'Stick Test (cm)': 15,
    'Strength Score': 25,
    'Endurance Score': 28,
    'Weekly Training Hours': 12,
    'Years Experience': 4,
    'Injury Duration (weeks)': 4,
    'Injury Occurred (weeks ago)': 8,
    'Coach': 1,
    'Gym Safety': 1,
    'Coach exp': 5,
    'Coach certification': 1,
    'coaches success %': 85,
    'current discomfort / Injury': 2
}

# Add sport one-hot encoding
sports = ['football', 'basketball', 'cricket', 'tennis', 'athletics', 'volleyball', 'swimming', 'badminton']
for sport in sports:
    test_data[f'sport_{sport}'] = 1 if sport == 'football' else 0

# Add injury location one-hot encoding
injury_locations = ['head', 'neck', 'shoulder', 'arm', 'elbow', 'knee', 'ankle', 'back', 'hip', 'foot']
for loc in injury_locations:
    test_data[f'previous_{loc}_injury'] = 1 if loc == 'shoulder' else 0

# Convert to DataFrame
input_data = pd.DataFrame([test_data])

print("\nInput data columns:")
print(input_data.columns.tolist())

# Make predictions
print("\nMaking predictions...")
try:
    severity_pred = severity_model.predict(input_data)
    print("Severity Prediction:", severity_pred)
except Exception as e:
    print("Error in severity prediction:", str(e))

try:
    location_pred = location_model.predict(input_data)
    location_name = location_encoder.inverse_transform([int(location_pred[0])])[0]
    print("Location Prediction:", location_name)
except Exception as e:
    print("Error in location prediction:", str(e))

# Check model attributes
print("\nSeverity Model Attributes:")
for attr in dir(severity_model):
    if not attr.startswith('_'):
        print(f"{attr}: {getattr(severity_model, attr, 'Not available')}")

print("\nLocation Model Attributes:")
for attr in dir(location_model):
    if not attr.startswith('_'):
        print(f"{attr}: {getattr(location_model, attr, 'Not available')}") 