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
    if request.method == 'POST':
        form_data = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Sport': request.form['sport'],
            'Weekly Training Hours': int(request.form['training_hours']),
            'Years of Experience': int(request.form['experience']),
            'Previous Injuries Count': int(request.form['previous_injuries']),
            'Average Warm-up Time': int(request.form['warmup_time']),
            'Rest Days per Week': int(request.form['rest_days'])
        }

        feature_columns = [
            'Age', 'Gender', 'Sport', 'Weekly Training Hours',
            'Years of Experience', 'Previous Injuries Count',
            'Average Warm-up Time', 'Rest Days per Week'
        ]
        input_df = pd.DataFrame([form_data], columns=feature_columns)
        input_df = pd.get_dummies(input_df, columns=['Gender', 'Sport'], drop_first=True)

        training_cols = [
            'Age', 'Weekly Training Hours', 'Years of Experience',
            'Previous Injuries Count', 'Average Warm-up Time', 'Rest Days per Week',
            'Gender_Male', 'Sport_Basketball', 'Sport_Football',
            'Sport_Running', 'Sport_Soccer', 'Sport_Tennis'
        ]
        input_df_aligned = input_df.reindex(columns=training_cols, fill_value=0)

        severity_prediction = severity_model.predict(input_df_aligned)[0]
        location_prediction_encoded = location_model.predict(input_df_aligned)[0]
        location_prediction = location_encoder.inverse_transform([location_prediction_encoded])[0]

        risk_score = severity_prediction * 2.5

        return render_template(
            'index.html',
            location=location_prediction,
            severity=int(severity_prediction),
            risk=round(risk_score, 1),
            form_data=form_data
        )

    return render_template('index.html')

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
