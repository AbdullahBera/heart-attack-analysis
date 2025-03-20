import streamlit as st
import pandas as pd 
import numpy as np
import joblib 
import shap 
import matplotlib.pyplot as plt 
import streamlit.components.v1 as components


model = joblib.load("models/best_model_rf.pkl")


tab1, tab2 = st.tabs(["BMI Calculator", "Heart Risk"])

with tab1: 
    st.header("BMI Calculator")

    bmi_height = st.number_input("Height (cm)", 140, 200, 170)
    bmi_weight = st.number_input("Weight (kg)", 40, 150, 70)

    bmi = round(bmi_weight / ((bmi_height / 100) ** 2), 2)

    st.write(f"Your BMI is: {bmi}")

with tab2: 
    st.title("Cardiovascular Disease Prediction")

    # User Inputs (matching dataset structure)
    age = st.slider("Age (in years)", 30, 80, 50)
    gender = st.selectbox("Gender", [1, 2])
    bmi = st.number_input("BMI", 12.0, 50.0, 22.5)
    ap_hi = st.number_input("Systolic BP", 90, 200, 120)
    ap_lo = st.number_input("Diastolic BP", 60, 120, 80)
    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
    gluc = st.selectbox("Glucose Level", [1, 2, 3])
    smoke = st.checkbox("Smoker")
    alco = st.checkbox("Alcohol Consumer")
    active = st.checkbox("Physically Active")

    # Prepare input data in the correct format
    input_data = pd.DataFrame([[age, gender, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi]],
                              columns=["age", "gender", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "bmi"])

    # Prediction
    prediction = model.predict(input_data)
    st.write(f"Predictions: {'Cardiovascular Disease' if prediction[0] == 1 else 'No Cardiovascular Disease'}")

    # Create SHAP explainer
    explainer = shap.Explainer(model)
    
    # Get SHAP values for the input
    shap_values = explainer(input_data)

    # Create a bar plot of feature importance
    st.subheader("Feature Impact Magnitude")
    
    # Get absolute SHAP values (magnitude of impact)
    feature_importance = np.abs(shap_values.values).mean(0)
    if len(feature_importance.shape) > 1:
        feature_importance = feature_importance.mean(1)
    
    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Plot with matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Feature Importance')
    
    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # For the individual prediction explanation, use a waterfall plot with matplotlib
    st.subheader("Explanation for Current Prediction")
    
    try:
        # Get the SHAP values for the first instance
        instance_shap_values = shap_values.values[0]
        
        # If SHAP values have 2 dimensions (for binary classification), 
        # take the values for the positive class (usually index 1)
        if len(instance_shap_values.shape) > 1 and instance_shap_values.shape[1] == 2:
            # Use values for the positive class (index 1)
            instance_shap_values = instance_shap_values[:, 1]
        elif len(instance_shap_values.shape) > 1:
            # If the shape is not exactly what we expect, just use the first row
            instance_shap_values = instance_shap_values[:, 0]
        
        # Get feature importance ranking
        sorted_idx = np.argsort(np.abs(instance_shap_values))
        sorted_features = np.array(input_data.columns)[sorted_idx]
        sorted_values = instance_shap_values[sorted_idx]
        
        # Plot top contributing features
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in sorted_values]
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Contributions to Current Prediction')
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add a legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Decreases prediction')
        green_patch = mpatches.Patch(color='green', label='Increases prediction')
        ax.legend(handles=[red_patch, green_patch])
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error generating feature contribution plot: {str(e)}")