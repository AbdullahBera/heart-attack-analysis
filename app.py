import streamlit as st
import pandas as pd 
import numpy as np
import joblib 
import shap 
import matplotlib.pyplot as plt 

model = joblib.load("models/best_model_rf.pkl")


tab1, tab2 = st.tabs(["BMI Calculator", "Heart Risk"])

with tab1: 
    st.header("BMI Calculator")

    height_cm = st.number_input("Height (cm)", 140, 200, 170)
    weight_kg = st.number_input("Weight (kg)", 40, 150, 70)

    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)

    st.write(f"Your BMI is: {bmi}")

with tab2: 
    st.title("Cardiovascular Disease Prediction")

    # User Inputs
    id_value = 0  # Placeholder since ID is not needed for predictions
    age = st.slider("Age", 30, 80, 50)

    # Height & Weight inputs to calculate BMI
    height_cm = st.number_input("Height (cm)", 140, 200, 170)
    weight_kg = st.number_input("Weight (kg)", 40, 150, 70)

    # Calculate BMI
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 2)

    # Blood Pressure
    ap_hi = st.number_input("Systolic BP", 90, 200, 120)
    ap_lo = st.number_input("Diastolic BP", 60, 120, 80)

    # Other Health Factors
    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
    gluc = st.selectbox("Glucose Level", [1, 2, 3])
    smoke = int(st.checkbox("Smoker"))  # Convert Boolean to int (0 or 1)
    alco = int(st.checkbox("Alcohol Consumer"))  # Convert Boolean to int (0 or 1)
    active = int(st.checkbox("Physically Active"))  # Convert Boolean to int (0 or 1)

    # Prepare input data
    input_data = np.array([[id_value, age, 1, height_cm, weight_kg, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])


    # Prediction
    prediction = model.predict(input_data)
    st.write(f"Predictions: {'Cardiovascular Disease' if prediction[0] == 1 else 'No Cardiovascular Disease'}")


    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values.values, input_data, matplotlib=True)
    st.pyplot(plt)
