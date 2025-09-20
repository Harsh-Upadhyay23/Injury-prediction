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
    # You might want to handle this more gracefully
    severity_model = location_model = location_encoder = None

@app.route('/')
def home():
    # Renders the initial form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # --- Collect all form data ---
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

        # --- Create a DataFrame for prediction ---
        # The order of columns must match the training data
        feature_columns = [
            'Age', 'Gender', 'Sport', 'Weekly Training Hours',
            'Years of Experience', 'Previous Injuries Count',
            'Average Warm-up Time', 'Rest Days per Week'
        ]
        input_df = pd.DataFrame([form_data], columns=feature_columns)

        # --- Preprocessing ---
        # Convert categorical variables to numerical using one-hot encoding
        input_df = pd.get_dummies(input_df, columns=['Gender', 'Sport'], drop_first=True)

        # Align columns with the model's training columns
        # This handles missing columns if a sport/gender wasn't in the form but was in training
        # We'll create a dummy training columns list for this example
        # In a real scenario, you'd save this from your training script
        # NOTE: This is a simplified list. Ensure it matches your actual model's features.
        training_cols = ['Age', 'Weekly Training Hours', 'Years of Experience',
                         'Previous Injuries Count', 'Average Warm-up Time', 'Rest Days per Week',
                         'Gender_Male', 'Sport_Basketball', 'Sport_Football',
                         'Sport_Running', 'Sport_Soccer', 'Sport_Tennis'] # Example columns
        
        # Reindex the input dataframe to match the training columns
        input_df_aligned = input_df.reindex(columns=training_cols, fill_value=0)


        # --- Make Predictions ---
        severity_prediction = severity_model.predict(input_df_aligned)[0]
        location_prediction_encoded = location_model.predict(input_df_aligned)[0]

        # Decode the location prediction
        location_prediction = location_encoder.inverse_transform([location_prediction_encoded])[0]

        # Calculate a risk score (example logic)
        risk_score = (severity_prediction * 2.5) # Scale severity (0-4) to a 1-10 score

        # --- Render results on the page ---
        return render_template('index.html',
                               location=location_prediction,
                               severity=int(severity_prediction),
                               risk=round(risk_score, 1),
                               form_data=form_data) # Pass original data back to the form

    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """
    This route handles the submission of actual injury data from the user.
    """
    if request.method == 'POST':
        # --- Collect the original input data passed through hidden fields ---
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
        
        # --- Collect the new, actual injury information ---
        injury_occurred = request.form.get('injury_occurred')
        
        # If an injury occurred, get the details. Otherwise, use placeholders.
        if injury_occurred == 'yes':
            actual_location = request.form['actual_location']
            actual_severity = request.form['actual_severity']
        else:
            actual_location = "No Injury"
            actual_severity = 0 # Assuming 0 means no injury

        # --- Prepare the new row for the CSV ---
        # The order must match your training data CSV for future retraining
        new_row = list(original_data.values()) + [actual_location, actual_severity]
        
        # --- Save the data to a CSV file ---
        file_path = os.path.join('data', 'new_data_to_add.csv')
        
        # Check if the file exists to determine if we need to write headers
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # If file doesn't exist, write headers first
                headers = list(original_data.keys()) + ['Injury Location', 'Injury Severity']
                writer.writerow(headers)
            writer.writerow(new_row)
            
        # --- Render the page again with a success message ---
        return render_template('index.html', submission_success=True)

    return render_template('index.html')

@app.route('/submit_athlete_data', methods=['POST'])
def submit_athlete_data():
    """
    Handle submission of athlete data from the new submit form.
    """
    try:
        # Collect all form data
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
        
        # Validate required fields
        required_fields = ['Name', 'Age', 'Gender', 'Sport']
        for field in required_fields:
            if not athlete_data[field]:
                return jsonify({'success': False, 'error': f'{field} is required'})
        
        # Save to CSV file
        file_path = os.path.join('data', 'athlete_submissions.csv')
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write headers if file doesn't exist
                writer.writerow(athlete_data.keys())
            writer.writerow(athlete_data.values())
        
        return jsonify({'success': True, 'message': 'Athlete data submitted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
