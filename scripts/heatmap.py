import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_feature_heatmap():
    """
    Loads the dataset, selects specified influential features,
    calculates their correlation matrix, and displays it as a heatmap.
    """
    print("Generating feature influence heatmap...")

    # Construct the path to the dataset
    # Assumes the script is run from the root directory of the project
    data_path = os.path.join('data', 'balanced_sheet1.csv')

    # Check if the data file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure the path is correct and the script is run from the project's root directory.")
        return

    # Load the dataset
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # List of the specific features you want to analyze
    # CORRECTED the feature names to match the likely CSV column headers
    influential_features = [
        'Injury Duration (weeks)',
        'Injury Occurred (weeks ago)',
        'BMI',
        'Weekly Training Hours',
        'current discomfort / Injury', # Corrected name
        'Weight (kg)',
        'Quad Circumference (cm)',
        'Wrist Circumference (cm)',
        'Ankle Circumference (cm)',
        'Trunk Flexion (cm)',
        'Shoulder Flexion (deg)',
        'Coach exp' # Corrected name
    ]

    # Verify that the requested columns exist in the DataFrame
    missing_features = [feature for feature in influential_features if feature not in df.columns]
    if missing_features:
        print("\nError: The following specified features were not found in the dataset:")
        for feature in missing_features:
            print(f"- {feature}")
        print("\nPlease check for typos or differences in column names.")
        # Tip: To see all available columns, uncomment the line below
        # print("\nAvailable columns:", df.columns.tolist())
        return

    # Create a new dataframe with only the influential features
    df_features = df[influential_features]

    # Calculate the correlation matrix for these features
    correlation_matrix = df_features.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(14, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap,
                linewidths=.5, cbar_kws={"shrink": .8})

    plt.title('Heatmap of Most Influential Features', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Adjust layout to make room for labels

    # Save the heatmap as an image file in the results directory
    output_path = os.path.join('results', 'feature_influence_heatmap.png')
    plt.savefig(output_path)
    print(f"\nHeatmap saved successfully to: {output_path}")

    # Display the heatmap
    plt.show()


if __name__ == '__main__':
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    generate_feature_heatmap()

