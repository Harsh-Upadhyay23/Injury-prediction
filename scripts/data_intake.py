import pandas as pd
import os

# File to store incoming user-submitted data
NEW_DATA_FILE = "data/new_data_to_add.csv"

# Columns required in incoming data
REQUIRED_COLUMNS = [
    'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
    'Waist Circumference (cm)', 'Hip Circumference (cm)', 'Quad Circumference (cm)',
    'Calf Circumference (cm)', 'Upper Arm Circumference (cm)', 'Wrist Circumference (cm)',
    'Ankle Circumference (cm)', 'Shoulder Flexion (deg)', 'Trunk Flexion (cm)', 'Stick Test (cm)',
    'Strength Score', 'Endurance Score', 'Weekly Training Hours', 'Years Experience',
    'Injury Location', 'Injury Severity'
]

def append_new_data(new_df: pd.DataFrame):
    # Check if all required columns are present
    if not all(col in new_df.columns for col in REQUIRED_COLUMNS):
        missing = list(set(REQUIRED_COLUMNS) - set(new_df.columns))
        raise ValueError(f"Missing columns in input: {missing}")

    # Append to CSV
    if os.path.exists(NEW_DATA_FILE):
        existing_df = pd.read_csv(NEW_DATA_FILE)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(NEW_DATA_FILE, index=False)
    print(f"✅ New data saved to {NEW_DATA_FILE}. Total rows now: {len(combined_df)}")

if __name__ == '__main__':
    # Example usage: simulate one row of input
    sample_input = pd.DataFrame([{
        'Age': 23, 'Gender': 1, 'Weight (kg)': 68, 'Height (m)': 1.75, 'BMI': 22.2,
        'Waist Circumference (cm)': 80, 'Hip Circumference (cm)': 90, 'Quad Circumference (cm)': 55,
        'Calf Circumference (cm)': 36, 'Upper Arm Circumference (cm)': 32, 'Wrist Circumference (cm)': 16,
        'Ankle Circumference (cm)': 22, 'Shoulder Flexion (deg)': 150, 'Trunk Flexion (cm)': 30,
        'Stick Test (cm)': 15, 'Strength Score': 75, 'Endurance Score': 85,
        'Weekly Training Hours': 12, 'Years Experience': 4,
        'Injury Location': 1, 'Injury Severity': 2
    }])

    append_new_data(sample_input)
