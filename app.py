import streamlit as st
import pandas as pd 
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

    # User Input 
    age = st.slider("Age", 30, 80, 50)
    bmi = bmi
    ap_hi = st.number_input("Systolic BP", 90, 200, 120)
    ap_lo = st.number_input("Diastolic BP", 60, 120, 80)
    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
    gluc = st.selectbox("Glucose Level", [1, 2, 3])
    smoke = st.checkbox("Smoker")
    alco = st.checkbox("Alcohol Consumer")
    active = st.checkbox("Physically Active")

    # Prediction
    input_data = [[age, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]
    prediction = model.prediction(input_data)
    st.write(f"Predictions: {'Cardiovascular Disease' if prediction[0] == 1 else 'No Cardiovascular Disease'}")


    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values.values, input_data, matplotlib=True)
    st.pyplot(plt)
