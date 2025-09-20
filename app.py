from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)

# Load the models and the encoder
try:
    severity_model = joblib.load('results/updated_advanced_hybrid/ensemble_2_(rf+xgb+nn)_severity_model.pkl')
    location_model = joblib.load('results/updated_advanced_hybrid/ensemble_4_(bagging)_location_model.pkl')
    location_encoder = joblib.load('results/models/location_encoder.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Make sure the paths are correct.")
    severity_model = location_model = location_encoder = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Create a dictionary from the form data
        # This uses the EXACT 'name' attributes from your index.html
        form_data = request.form.to_dict()

        # Step 2: Convert necessary fields from string to number
        # Also handles cases where optional fields might be empty
        numeric_features = {
            'Age': float(form_data.get('Age', 0)),
            'Weight (kg)': float(form_data.get('Weight (kg)', 0)),
            'Height (m)': float(form_data.get('Height (m)', 0)),
            'BMI': float(form_data.get('BMI', 0)),
            'Waist Circumference (cm)': float(form_data.get('Waist Circumference (cm)', 66.45)),
            'Hip Circumference (cm)': float(form_data.get('Hip Circumference (cm)', 82.88)),
            'Quad Circumference (cm)': float(form_data.get('Quad Circumference (cm)', 51.52)),
            'Calf Circumference (cm)': float(form_data.get('Calf Circumference (cm)', 32.81)),
            'Upper Arm Circumference (cm)': float(form_data.get('Upper Arm Circumference (cm)', 26.92)),
            'Wrist Circumference (cm)': float(form_data.get('Wrist Circumference (cm)', 15.91)),
            'Ankle Circumference (cm)': float(form_data.get('Ankle Circumference (cm)', 18.58)),
            'Shoulder Flexion (deg)': float(form_data.get('Shoulder Flexion (deg)', 180.30)),
            'Trunk Flexion (cm)': float(form_data.get('Trunk Flexion (cm)', 5.00)),
            'Stick Test (cm)': float(form_data.get('Stick Test (cm)', 24.99)),
            'Strength Score': float(form_data.get('Strength Score', 3.06)),
            'Endurance Score': float(form_data.get('Endurance Score', 3.10)),
            'Training hrs': float(form_data.get('Training hrs', 0)),
            'Experience': float(form_data.get('Experience', 0)),
            'Duration': float(form_data.get('Duration', 0)),
            'Injury Occurred (weeks ago)': float(form_data.get('Injury Occurred (weeks ago)', 9.09)),
            'discomfort': float(form_data.get('discomfort', 0)),
            'Gym Safety': float(form_data.get('Gym Safety', 0))
        }

        # Add categorical feature
        numeric_features['Gender'] = form_data.get('Gender', 'Male')

        # Step 3: Create a DataFrame in the correct order for the model
        # THIS ORDER MUST EXACTLY MATCH YOUR MODEL'S TRAINING DATA
        feature_order = [
            'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
            'Waist Circumference (cm)', 'Hip Circumference (cm)',
            'Quad Circumference (cm)', 'Calf Circumference (cm)',
            'Upper Arm Circumference (cm)', 'Wrist Circumference (cm)',
            'Ankle Circumference (cm)', 'Shoulder Flexion (deg)',
            'Trunk Flexion (cm)', 'Stick Test (cm)', 'Strength Score',
            'Endurance Score', 'Training hrs', 'Experience', 'Duration',
            'Injury Occurred (weeks ago)', 'discomfort', 'Gym Safety'
        ]
        
        input_df = pd.DataFrame([numeric_features], columns=feature_order)

        # Pre-processing (handle categorical data)
        input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

        # --- Make Predictions ---
        severity_prediction = severity_model.predict(input_df)[0]
        location_prediction_encoded = location_model.predict(input_df)[0]
        location_prediction = location_encoder.inverse_transform([location_prediction_encoded])[0]

        # Calculate risk score (0-100 scale)
        # Assuming severity is on a scale, e.g., 1-4. Adjust max_severity if different.
        max_severity = 4 
        risk_score = (severity_prediction / max_severity) * 100

        # Step 4: Return a JSON response
        return jsonify({
            'risk_score': round(risk_score, 2),
            'severity': str(int(severity_prediction)), # Convert numpy int to standard int
            'location': location_prediction
        })

    except Exception as e:
        # If any error occurs, return it as JSON
        return jsonify({'error': str(e)}), 400

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        original_data = {
            'Age': request.form['original_Age'],
            'Gender': request.form['original_Gender'],
            'Sport': request.form['original_Sport'],
            'Weekly Training Hours': request.form['original_Weekly Training Hours'],
            'Years of Experience': request.form['original_Years of Experience'],
            'Previous Injuries Count': request.form['original_Previous Injuries Count'],
            'Average Warm-up Time': request.form['original_Average Warm-up Time'],
            'Rest Days per Week': request.form['original_Rest Days per Week']
        }

        injury_occurred = request.form.get('injury_occurred')
        if injury_occurred == 'yes':
            actual_location = request.form['actual_location']
            actual_severity = request.form['actual_severity']
        else:
            actual_location = "No Injury"
            actual_severity = 0

        new_row = list(original_data.values()) + [actual_location, actual_severity]

        file_path = os.path.join('data', 'new_data_to_add.csv')
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = list(original_data.keys()) + ['Injury Location', 'Injury Severity']
                writer.writerow(headers)
            writer.writerow(new_row)

        return render_template('index.html', submission_success=True)

    return render_template('index.html')

@app.route('/submit_athlete_data', methods=['POST'])
def submit_athlete_data():
    try:
        athlete_data = {
            'Name': request.form.get('athlete_name', ''),
            'Age': request.form.get('athlete_age', ''),
            'Gender': request.form.get('athlete_gender', ''),
            'Sport': request.form.get('athlete_sport', ''),
            'Injury Duration (weeks)': request.form.get('injury_duration_weeks', ''),
            'Injury Occurred (weeks ago)': request.form.get('injury_occurred_weeks', ''),
            'Trunk Flexion (cm)': request.form.get('trunk_flexion_cm', ''),
            'BMI': request.form.get('athlete_bmi', ''),
            'Weekly Training Hours': request.form.get('weekly_training_hours', ''),
            'Current discomfort / Injury': request.form.get('current_discomfort', ''),
            'Shoulder Flexion (deg)': request.form.get('shoulder_flexion_deg', ''),
            'Weight (kg)': request.form.get('athlete_weight', ''),
            'Position': request.form.get('position', ''),
            'Submission Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        required_fields = ['Name', 'Age', 'Gender', 'Sport']
        for field in required_fields:
            if not athlete_data[field]:
                return jsonify({'success': False, 'error': f'{field} is required'})

        file_path = os.path.join('data', 'athlete_submissions.csv')
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(athlete_data.keys())
            writer.writerow(athlete_data.values())

        return jsonify({'success': True, 'message': 'Athlete data submitted successfully'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Render-ready startup
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
