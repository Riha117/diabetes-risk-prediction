
import streamlit as st
import pandas as pd
import joblib
import sqlite3

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f8fbff, #e6f0ff);
    }

    h1 {
        color: #0A66C2;
    }

    .stButton>button {
        background-color: #0A66C2;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }

    .stSlider {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

st.markdown("## Enter Health Details")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.slider(
        "Pregnancies",
        0, 20, 1
    )

    glucose = st.slider(
        "Glucose Level",
        50, 250, 100
    )

    blood_pressure = st.slider(
        "Blood Pressure",
        40, 180, 80
    )

    skin_thickness = st.slider(
        "Skin Thickness",
        0, 100, 20
    )

with col2:

    insulin = st.slider(
        "Insulin Level",
        0, 900, 80
    )

    bmi = st.slider(
        "BMI",
        10.0, 60.0, 25.0
    )

    dpf = st.slider(
        "Diabetes Pedigree Function",
        0.0, 3.0, 0.5
    )

    age = st.slider(
        "Age",
        1, 100, 25
    )
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
