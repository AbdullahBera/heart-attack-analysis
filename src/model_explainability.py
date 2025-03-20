import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier


def explain_model(model: RandomForestClassifier, X_test: pd.DataFrame) -> str:
    """
    Generates a SHAP summary plot to explain the model's predictions.

    Parameters:
    model_path : str
        The file path to the trained model (e.g., "best_model_rf.pkl").
    X_test : pd.DataFrame
        The feature matrix for test data.

    Returns:
    plot_path : str
        The file path where the SHAP summary plot is saved.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    plt.savefig("models/shap_summary.png")


if __name__ == "__main__":

    model = joblib.load("models/best_model_rf.pkl")
    X_test = pd.read_csv("data/X_test.csv")

    explain_model(model, X_test)
