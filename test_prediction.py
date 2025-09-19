import requests

# Test data
data = {
    'Age': '25',
    'Gender': 'Male',
    'Weight (kg)': '70',
    'Height (m)': '1.75',
    'BMI': '22.86',
    'Training hrs': '5',
    'Experience': '3',
    'Duration': '14',
    'discomfort': '3',
    'Gym Safety': '8'
}

# Send POST request
response = requests.post('http://127.0.0.1:5000/predict', data=data)
print("Response status code:", response.status_code)
print("Response content:", response.json()) 