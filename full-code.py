# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# Load dataset and perform data preprocessing
def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Basic data checks
    print("Dataset Info:")
    print(df.info())
    
    # Split dataset into features and target variable
    X = df.drop('target_column', axis=1)  # Replace with your actual target column
    y = df['target_column']  # Replace with your actual target column
    
    # Train-test split (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Exploratory Data Analysis (EDA)
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

# Train and evaluate the model
def train_models(X_train, X_test, y_train, y_test):
    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# Hyperparameter tuning using GridSearchCV
def tune_models(X_train, X_test, y_train, y_test):
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30, None],
        'criterion': ['gini', 'entropy']
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Display best parameters
    print("Best Hyperparameters:", grid_search.best_params_)

# Save trained model
def save_model(model, filename):
    # Save the trained model
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Main function to run the entire pipeline
def main():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('your_dataset.csv')  # Replace with actual file path
    
    # Perform exploratory data analysis
    perform_eda(pd.read_csv('your_dataset.csv'))  # Replace with actual file path
    
    # Train models and evaluate performance
    train_models(X_train, X_test, y_train, y_test)

    # Hyperparameter tuning
    tune_models(X_train, X_test, y_train, y_test)
    
    # Example: Save the final model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    save_model(model, 'final_model.pkl')

# Run the main function
if __name__ == "__main__":
    main()
