import streamlit as st
import pickle
import numpy as np

st.title("🩺 Diabetes Risk Predictor")

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Input form
st.write("Enter your health details:")

pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = "High risk of Diabetes" if prediction[0] == 1 else "Low risk of Diabetes"
    st.success(f"Result: **{result}**")
