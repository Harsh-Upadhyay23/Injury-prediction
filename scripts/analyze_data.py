import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(filepath):
    # Read the dataset
    df = pd.read_excel(filepath)
    
    print("\n=== Dataset Overview ===")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    print("\n=== Column Information ===")
    print(df.info())
    
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    print("\n=== Target Variables Distribution ===")
    if 'Injury Severity' in df.columns:
        print("\nInjury Severity Distribution:")
        print(df['Injury Severity'].value_counts())
        
    if 'Injury Location' in df.columns:
        print("\nInjury Location Distribution:")
        print(df['Injury Location'].value_counts())
    
    # Save the analysis to a text file
    with open('results/dataset_analysis.txt', 'w') as f:
        f.write("=== Dataset Analysis Report ===\n\n")
        f.write(f"Number of rows: {len(df)}\n")
        f.write(f"Number of columns: {len(df.columns)}\n\n")
        f.write("=== Column Information ===\n")
        df.info(buf=f)
        f.write("\n\n=== Basic Statistics ===\n")
        f.write(str(df.describe()))
        f.write("\n\n=== Missing Values ===\n")
        f.write(str(missing_values[missing_values > 0]))
        f.write("\n\n=== Target Variables Distribution ===\n")
        if 'Injury Severity' in df.columns:
            f.write("\nInjury Severity Distribution:\n")
            f.write(str(df['Injury Severity'].value_counts()))
        if 'Injury Location' in df.columns:
            f.write("\nInjury Location Distribution:\n")
            f.write(str(df['Injury Location'].value_counts()))

if __name__ == "__main__":
    # Analyze original dataset
    print("Analyzing original dataset...")
    analyze_dataset("data/sheet1.xlsx")
    
    # Analyze balanced dataset
    print("\nAnalyzing balanced dataset...")
    analyze_dataset("data/balanced_sheet1.xlsx") 