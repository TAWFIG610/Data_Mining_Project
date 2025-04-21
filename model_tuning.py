# model_tuning.py

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_models(df):
    # Define features and target
    X = df.drop('target_column', axis=1)
    y = df['target_column']
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30, None],
        'criterion': ['gini', 'entropy']
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Display best parameters
    print("Best Hyperparameters:", grid_search.best_params_)
