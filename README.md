# Sports Injury Risk Predictor

A web application that predicts the risk of sports injuries based on various factors. The application uses advanced machine learning models to predict both the location and severity of potential injuries.

## Features

- Predicts most likely injury location
- Estimates injury severity level
- Provides a risk score (1-10)
- User-friendly web interface
- Real-time predictions

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the trained models are in the correct location:
- `results/advanced_hybrid/ensemble_2_rf_xgb_nn_severity_model.pkl`
- `results/advanced_hybrid/ensemble_4_bagging_location_model.pkl`
- `results/models/location_encoder.pkl`

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Fill in the form with your details:
   - Age
   - Gender
   - Sport
   - Weekly Training Hours
   - Years of Experience
   - Previous Injuries Count
   - Average Warm-up Time
   - Rest Days per Week

2. Click "Predict Injury Risk" to get your results

3. View your prediction results:
   - Most likely injury location
   - Severity level
   - Risk score (1-10)

## Model Information

- Severity Prediction: Ensemble Model (RF+XGB+NN) with 92.72% accuracy
- Location Prediction: Bagging Ensemble with 78.15% accuracy

## Notes

- The risk score is calculated on a scale of 1-10, with 10 being the highest risk
- Severity levels range from 0-4, with 4 being the most severe
- The application uses advanced ensemble models based on historical sports injury data
- The models combine multiple algorithms to provide more accurate predictions:
  - For severity: Random Forest, XGBoost, and Neural Network
  - For location: Bagging with XGBoost as base estimator 