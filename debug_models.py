#!/usr/bin/env python3
"""
Debug the models to see what's happening
"""

import joblib
import pandas as pd
import numpy as np

def debug_models():
    """Debug the best models"""
    
    print("🔍 Debugging Best Models...")
    print("=" * 40)
    
    # Load models
    try:
        severity_model = joblib.load("results/best_models/best_severity_model.pkl")
        location_model = joblib.load("results/best_models/best_location_model.pkl")
        location_encoder = joblib.load("results/best_models/location_encoder.pkl")
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Create test data exactly as Flask app does
    input_data = {
        'Age': 25,
        'Gender': 1,
        'Weight (kg)': 70,
        'Height (m)': 1.75,
        'BMI': 22.9,
        'Waist Circumference (cm)': 85,
        'Hip Circumference (cm)': 95,
        'Quad Circumference (cm)': 55,
        'Calf Circumference (cm)': 35,
        'Upper Arm Circumference (cm)': 32,
        'Wrist Circumference (cm)': 17,
        'Ankle Circumference (cm)': 22,
        'Shoulder Flexion (deg)': 160,
        'Trunk Flexion (cm)': 15,
        'Stick Test (cm)': 15,
        'Strength Score': 7,
        'Endurance Score': 7,
        'Weekly Training Hours': 5,
        'Years Experience': 3,
        'Injury Duration (weeks)': 2,
        'Injury Occurred (weeks ago)': 8,
        'Unnamed: 23': 0,
        'Coach': 1,
        'Gym Safety': 8,
        'Coach exp': 5,
        'Coach certification': 1,
        'coaches success %': 75,
        'current discomfort / Injury': 3
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    print(f"Input DataFrame shape: {input_df.shape}")
    print(f"Input DataFrame columns: {list(input_df.columns)}")
    
    # Test severity model
    try:
        print("\n🔍 Testing Severity Model...")
        severity_pred = severity_model.predict(input_df)
        severity_proba = severity_model.predict_proba(input_df)
        print(f"Severity prediction: {severity_pred}")
        print(f"Severity probabilities: {severity_proba}")
        print(f"Severity prediction type: {type(severity_pred)}")
        print(f"Severity prediction shape: {severity_pred.shape}")
    except Exception as e:
        print(f"❌ Severity model error: {e}")
        return
    
    # Test location model
    try:
        print("\n🔍 Testing Location Model...")
        location_pred = location_model.predict(input_df)
        location_proba = location_model.predict_proba(input_df)
        print(f"Location prediction: {location_pred}")
        print(f"Location probabilities: {location_proba}")
        print(f"Location prediction type: {type(location_pred)}")
        print(f"Location prediction shape: {location_pred.shape}")
        
        # Test encoder
        location_name = location_encoder.inverse_transform(location_pred)
        print(f"Location name: {location_name}")
    except Exception as e:
        print(f"❌ Location model error: {e}")
        return
    
    print("\n✅ All models working correctly!")

if __name__ == "__main__":
    debug_models()
