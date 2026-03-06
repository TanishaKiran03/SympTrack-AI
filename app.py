import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("models/rf_model.pkl")

# Load dataset to get symptom list
data = pd.read_csv("data/Training.csv")
symptoms = data.columns[:-1]

st.title("🩺 SympTrack AI")
st.write("Select your symptoms to predict possible disease")

# Symptom selection
selected_symptoms = st.multiselect(
    "Choose Symptoms",
    symptoms
)

# Create input vector
input_vector = np.zeros(len(symptoms))

for symptom in selected_symptoms:
    index = list(symptoms).index(symptom)
    input_vector[index] = 1

# Predict button
if st.button("Predict Disease"):

    prediction = model.predict([input_vector])[0]
    probability = max(model.predict_proba([input_vector])[0])

    st.success(f"Predicted Disease: {prediction}")
    st.info(f"Confidence: {probability:.2f}")