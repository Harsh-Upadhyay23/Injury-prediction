#!/usr/bin/env python3
"""
Test the Flask app to ensure it's working correctly
"""

import requests
import json

def test_flask_app():
    """Test the Flask app with sample data"""
    
    print("🧪 Testing Flask App...")
    print("=" * 40)
    
    # Test data matching the form fields
    test_data = {
        'Age': '25',
        'Gender': 'Male',
        'Weight (kg)': '70',
        'Height (m)': '1.75',
        'BMI': '22.9',
        'Training hrs': '5',
        'Experience': '3',
        'Duration': '14',  # 14 days
        'Gym Safety': '8',
        'discomfort': '3'
    }
    
    try:
        # Test the home page
        print("1. Testing home page...")
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print("   ✅ Home page loads successfully")
        else:
            print(f"   ❌ Home page error: {response.status_code}")
            return
        
        # Test the prediction endpoint
        print("2. Testing prediction endpoint...")
        response = requests.post('http://localhost:5000/predict', data=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Prediction successful!")
            print(f"   Severity: {result.get('severity')}")
            print(f"   Location: {result.get('location')}")
            print(f"   Risk Score: {result.get('risk_score')}")
        else:
            print(f"   ❌ Prediction error: {response.status_code}")
            print(f"   Response: {response.text}")
            return
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask app. Make sure it's running on localhost:5000")
        return
    except Exception as e:
        print(f"❌ Error testing Flask app: {e}")
        return
    
    print("\n🎉 Flask app is working correctly!")
    print("You can now access it at: http://localhost:5000")

if __name__ == "__main__":
    test_flask_app()
