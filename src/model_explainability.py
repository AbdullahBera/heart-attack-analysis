import pandas as pd
import numpy as np 
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


    # Plot SHAP Summary (Overall Feature Importance)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    summary_plot_path = "models/shap_summary.png"
    plt.savefig(summary_plot_path)
    plt.close()

    # Compute SHAP Interaction Values
    shap_interaction_values = explainer.shap_interaction_values(X_test)

    # Find Top 5 Features
    interaction_values = np.abs(shap_interaction_values).mean(axis=0)
    top_features_idx = np.argsort(interaction_values.sum(axis=1))[::-1][:5]  
    feature_names = X_test.columns
    top_features = feature_names[top_features_idx]

    print(f"Top 5 most important features: {list(top_features)}")

    # Plot Feature Interactions (Age vs Other Features)
    plt.figure()
    shap.dependence_plot(
        top_features[0], shap_values.values, X_test, interaction_index=top_features[1]
    )
    interaction_plot_path = "models/shap_interaction.png"
    plt.savefig(interaction_plot_path)
    plt.close()

    print(f"SHAP summary plot saved at: {summary_plot_path}")
    print(f"Feature interaction plot saved at: {interaction_plot_path}")

    return summary_plot_path, interaction_plot_path



if __name__ == "__main__":

    model = joblib.load("models/best_model_rf.pkl")
    X_test = pd.read_csv("data/X_test.csv")

    explain_model(model, X_test)
