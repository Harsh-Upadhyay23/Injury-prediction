import pandas as pd

def load_and_clean_data(filepath: str, target_column: str):
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, engine='openpyxl')
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    df.dropna(subset=[target_column], inplace=True)

    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    numerical_columns = df.select_dtypes(include=['number']).columns

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    for column in non_numeric_columns:
        if not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode()[0])

    # Ensure 'Gender' column is properly handled
    if 'Gender' in df.columns:
        gender_mode = df['Gender'].mode()
        if not gender_mode.empty:
            df['Gender'] = df['Gender'].fillna(gender_mode[0])
        else:
            df['Gender'] = 'Unknown'  # Default value if mode is empty

    # Drop columns with all missing values
    df.dropna(axis=1, how='all', inplace=True)

    if df.isnull().sum().sum() > 0:
        print("Columns with missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        raise ValueError("Data still contains missing values after preprocessing.")

    return df

def encode_categorical_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df
