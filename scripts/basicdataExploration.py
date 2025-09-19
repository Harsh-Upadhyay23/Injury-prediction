# Load essential libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel('balanced_sheet1.xlsx')

# Overview of the dataset
print("Shape of the dataset:", df.shape)
print("Dataset Info:")
print(df.info())
print("Dataset Head:")
print(df.head())
print("Missing Value Counts:")
print(df.isnull().sum())
print("Statistical Summary of the Dataset:")
print(df.describe())

# Handle missing values in Gender column
if 'Gender' in df.columns:
    if df['Gender'].dtype == 'object':  # Only encode if categorical
        gender_mode = df['Gender'].mode()
        if not gender_mode.empty:
            df['Gender'] = df['Gender'].fillna(gender_mode[0])
        else:
            df['Gender'] = 'Unknown'
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'Unknown': -1})
    else:
        df['Gender'] = df['Gender'].fillna(df['Gender'].mean())  # If numeric, use mean

# Plot histograms for numerical columns
df.hist(bins=15, figsize=(20, 15))

# Apply fine-tuned subplot adjustments
plt.subplots_adjust(left=0.026, bottom=0.074, right=0.992, top=0.971, wspace=0.16, hspace=0.59)
plt.show()

# Select only numerical columns for correlation
numerical_df = df.select_dtypes(include=['number'])

# Compute and display the correlation matrix
correlation_matrix = numerical_df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Plot heatmap for better visualization
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Visualize missing values using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()

# Handle missing values
target_column = 'Injury Severity'  # Replace with your actual target column name

if target_column in df.columns:
    df.dropna(subset=[target_column], inplace=True)  # Drop rows with missing target values

# Identify non-numeric columns to avoid issues
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_columns)

# Impute missing values only for numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Impute missing values for categorical columns (optional)
categorical_columns = df.select_dtypes(exclude=['number']).columns
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])  # Impute using mode for categorical columns

print("Final cleaned and scaled dataset preview:")
print(df.head())

# Exploratory Data Analysis (EDA)

# Plot target variable distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Injury Duration (weeks)'], kde=True)
plt.title("Target Variable Distribution: Injury Duration")
plt.show()

# Detect outliers using boxplots for key features
key_features = ['Age', 'Weight (kg)', 'Height (m)', 'Strength Score', 'Endurance Score']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot for {feature}")
plt.tight_layout()
plt.show()

# Relationship between 'Strength Score' and 'Injury Duration'
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Strength Score'], y=df['Injury Duration (weeks)'])
plt.title("Strength Score vs Injury Duration")
plt.show()

# Feature Extraction and Identification of Non-required Data
print("\nStarting Feature Extraction...")

# Calculate correlation to the target variable if applicable
correlation_threshold = 0.8  # Define a threshold for high correlation

# Ensure only numerical columns are used for correlation
if target_column in df.columns:
    # Select numerical columns
    numerical_features = df.select_dtypes(include=['number']).columns
    if target_column in numerical_features:
        correlation_matrix = df[numerical_features].corr()
        highly_correlated_features = correlation_matrix[target_column][
            abs(correlation_matrix[target_column]) > correlation_threshold
        ].index
        print("Highly correlated features (correlation > 0.8):", list(highly_correlated_features))

# Drop irrelevant or highly correlated features (example shown below)
columns_to_drop = ['Waist Circumference', 'Coach Experience Years']  # Replace as needed
df = df.drop(columns=columns_to_drop, errors='ignore')

print("Remaining Columns After Feature Extraction:")
print(df.columns)

# Save the cleaned dataset for further modeling steps
df.to_csv('cleaned_dataset.csv', index=False)
print("Cleaned dataset saved successfully.")

import pandas as pd

def load_and_clean_data(filepath, target_column):
    df = pd.read_csv(filepath)
    # Data cleaning steps here
    return df
