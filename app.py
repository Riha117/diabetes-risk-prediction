
import streamlit as st
import pandas as pd
import joblib
import sqlite3

if "history" not in st.session_state:
    st.session_state.history = []
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)
# Load trained model
model = joblib.load("diabetes_model.pkl")

# Title
st.title("Diabetes Risk Prediction App")

st.write("Enter health details below:")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1)
# Predict Button
if st.button("Predict Diabetes Risk"):

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
    risk = "High" if prediction == 1 else "Low"

    probability = model.predict_proba(input_data)[0][1]
    
    st.session_state.history.append({
    "Age": age,
    "Glucose": glucose,
    "BMI": bmi,
    "Risk": risk
})

    st.subheader("Prediction Result")
  
    st.subheader("Prediction Probability")

    st.progress(float(probability))

    st.write(f"Risk Probability: {probability:.2%}")

    if prediction == 1:
        st.error(f"High Diabetes Risk ({probability:.2%})")
    else:
        st.success(f"Low Diabetes Risk ({probability:.2%})")
st.subheader("Prediction History")

history_df = pd.DataFrame(st.session_state.history)

st.dataframe(history_df)
