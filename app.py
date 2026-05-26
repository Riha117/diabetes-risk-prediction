import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import plotly.graph_objects as go

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b,
            #0f766e
        );
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: white;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #cbd5e1;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 15px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 15px;
        height: 3em;
        font-size: 20px;
        background: linear-gradient(
            90deg,
            #06b6d4,
            #3b82f6
        );
        color: white;
        border: none;
    }

    .stSlider {
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .glass {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
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
st.markdown(
    """
    <div class="main-title">
    🩺 AI Diabetes Prediction System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Advanced Healthcare Analytics Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
selected = option_menu(
    menu_title=None,
    options=["Prediction", "Analytics", "About"],
    icons=["activity", "bar-chart", "info-circle"],
    orientation="horizontal"
)
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

    if st.button("Predict Diabetes Risk"):

        input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]]

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    fig = go.Figure(go.Indicator(
    mode = "gauge+number",

    value = probability * 100,

    title = {'text': "Diabetes Risk"},

    gauge = {
        'axis': {'range': [0, 100]},

        'bar': {'color': "red"},

        'steps': [
            {'range': [0, 50], 'color': "lightgreen"},
            {'range': [50, 80], 'color': "orange"},
            {'range': [80, 100], 'color': "red"}
        ]
    }
))

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Glucose", glucose)

with col2:
    st.metric("BMI", bmi)

with col3:
    st.metric("Age", age)

chart_data = pd.DataFrame({
    "Health Metrics": [
        "Glucose",
        "BMI",
        "Blood Pressure"
    ],
    "Values": [
        glucose,
        bmi,
        blood_pressure
    ]
})

st.bar_chart(
    chart_data.set_index("Health Metrics")
)

tab1, tab2 = st.tabs([
    "Prediction",
    "Health Tips"
])

with tab1:
    st.write("Prediction dashboard")

with tab2:
    st.info(
        """
        ✔ Exercise regularly
        
        ✔ Maintain healthy BMI
        
        ✔ Reduce sugar intake
        
        ✔ Monitor glucose levels
        """
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Glucose", glucose)

with col2:
    st.metric("BMI", bmi)

with col3:
    st.metric("Age", age)
st.plotly_chart(fig)

st.subheader("Prediction Result")

st.progress(float(probability))

    if prediction == 1:
        st.error(
            f"⚠ High Diabetes Risk ({probability:.2%})"
        )
    else:
        st.success(
            f"✅ Low Diabetes Risk ({probability:.2%})"
        )
st.subheader("Prediction History")

history_df = pd.DataFrame(st.session_state.history)

st.dataframe(history_df)
