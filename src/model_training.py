import pandas as pd 
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    '''
    Train data using Random Forest using cross-validation to 
    find the best hyperparameters  

    Parameters: 
    X_train: pd.DataFrame
        The training feature matrix (row=samples, columns=features)

    y_train: pd.Series
        The target variable (1D array of class labels)

    Returns: 
    best_model = RandomForestClassifier
        The best-trained Random Forest model based on cross validation
    '''

    # Define the hyperparameters for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Train the model using Random Forest and GridSearchCV
    rf = RandomForestClassifier(random_state=19)
    grid_search = GridSearchCV(rf, 
                               param_grid, 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=2)

    # Fit the model using Grid search 
    grid_search.fit(X_train, y_train)

    print(f"Best Params: {grid_search.best_params_}")

    # Retrives the best model
    best_model = grid_search.best_estimator_

    return best_model


if __name__ == "__main__":
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()


    best_model = train_model(X_train, y_train)

    joblib.dump(best_model, "models/best_model_rf.pkl")

    print("Best model saved as best_model_rf.pkl")



