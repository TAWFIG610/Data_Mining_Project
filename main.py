# main.py

from data_preprocessing import load_and_preprocess_data
from exploratory_data_analysis import perform_eda
from model_training import train_models
from model_tuning import tune_models

def main():
    # Load and preprocess the data
    df = load_and_preprocess_data('your_dataset.csv')
    
    # Perform exploratory data analysis
    perform_eda(df)

    # Train models and evaluate performance
    train_models(df)

    # Hyperparameter tuning
    tune_models(df)

if __name__ == "__main__":
    main()
