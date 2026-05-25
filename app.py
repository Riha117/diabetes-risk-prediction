
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load trained model
model = joblib.load("diabetes_model.pkl")

# Title
st.markdown(
    "<h1 style='text-align:center;'>🩺 Diabetes Risk Predictor</h1>",
    unsafe_allow_html=True
)
st.sidebar.title("About")

st.sidebar.info(
    """
    This app predicts diabetes risk using a Random Forest Machine Learning model.
    
    Built with:
    - Python
    - Scikit-learn
    - Streamlit
    """
)

st.write("Enter health details below:")

# User Inputs
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)

glucose = st.number_input("Glucose Level", min_value=0)

blood_pressure = st.number_input("Blood Pressure", min_value=0)

skin_thickness = st.number_input("Skin Thickness", min_value=0)

insulin = st.number_input("Insulin Level", min_value=0)

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

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Diabetes Risk ({probability:.2%})")
    else:
        st.success(f"Low Diabetes Risk ({probability:.2%})")
