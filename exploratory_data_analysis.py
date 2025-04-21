# exploratory_data_analysis.py

import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):
    # Exploratory Data Analysis
    print("Summary Statistics:")
    print(df.describe())
    
    # Missing Values Heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cmap="coolwarm", cbar=False, yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.show()
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Distribution of GPA
    sns.histplot(df['GPA'], kde=True, bins=30)
    plt.title("GPA Distribution")
    plt.xlabel("GPA")
    plt.show()

    # Scatter Plot for GPA vs Other Numerical Features
    numerical_columns = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']
    for num_col in numerical_columns:
        plt.scatter(df[num_col], df['GPA'], alpha=0.5)
        plt.title(f"Scatter Plot: {num_col} vs GPA")
        plt.xlabel(num_col)
        plt.ylabel("GPA")
        plt.show()
