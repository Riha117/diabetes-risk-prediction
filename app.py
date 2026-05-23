import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# Title
st.title("Diabetes Risk Prediction")

st.write("Enter patient details below:")

# Inputs
pregnancies = st.number_input("Pregnancies", min_value=0)

glucose = st.number_input("Glucose", min_value=0)

blood_pressure = st.number_input("Blood Pressure", min_value=0)

skin_thickness = st.number_input("Skin Thickness", min_value=0)

insulin = st.number_input("Insulin", min_value=0)

bmi = st.number_input("BMI", min_value=0.0)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=1)

# Predict
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Diabetes Risk ({probability:.2%})")
    else:
        st.success(f"Low Diabetes Risk ({probability:.2%})")
