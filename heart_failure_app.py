
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("heart_failure_model.pkl")

st.title("Heart Failure Risk Predictor")

st.write("Enter the patient's clinical data:")

# Input fields for all 12 features
age = st.number_input("Age", min_value=1, max_value=120, value=60)
anaemia = st.selectbox("Anaemia (0 = No, 1 = Yes)", [0, 1])
cpk = st.number_input("Creatinine Phosphokinase", value=250)
diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
ef = st.slider("Ejection Fraction (%)", min_value=10, max_value=80, value=38)
hbp = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
platelets = st.number_input("Platelets", value=250000)
sc = st.number_input("Serum Creatinine", value=1.2)
ss = st.number_input("Serum Sodium", value=137)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
time = st.number_input("Follow-up Time (days)", value=100)

# Make prediction
if st.button("Predict"):
    data = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets,
                      sc, ss, sex, smoking, time]])
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]  # Probability of death

    if prediction == 1:
        st.error(f"Prediction: High Risk of Death ({proba*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: Likely to Survive ({(1-proba)*100:.2f}% confidence)")
