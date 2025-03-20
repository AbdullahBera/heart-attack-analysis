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

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    # Convert input_data to a 1D array
    input_data_array = input_data.iloc[0].values  

    # Ensure expected_value is a single number
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0]

    # Ensure SHAP values are 1D
    shap_values_array = shap_values.values[0]

    # Generate SHAP force plot
    force_plot_html = shap.force_plot(expected_value, shap_values_array, input_data_array, matplotlib=False)

    # Render SHAP plot in Streamlit using HTML
    components.html(shap.getjs(), height=0)  # Load SHAP JS
    components.html(force_plot_html.html(), height=300)
