import streamlit as st
import pickle
import numpy as np


# Load model
with open("wine.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Title
st.title("🍷 Wine Type Prediction")

st.write("Enter the wine characteristics to predict the wine type.")


# Input fields
alcohol = st.number_input(
    "Alcohol",
    min_value=0.0,
    value=13.0
)

malic_acid = st.number_input(
    "Malic Acid",
    min_value=0.0,
    value=2.0
)

ash = st.number_input(
    "Ash",
    min_value=0.0,
    value=2.3
)

phenols = st.number_input(
    "Phenols",
    min_value=0.0,
    value=2.0
)

flavanoids = st.number_input(
    "Flavanoids",
    min_value=0.0,
    value=2.0
)


# Prediction button
if st.button("Predict Wine Type"):

    # Create input array
    features = np.array([
        [alcohol, malic_acid, ash, phenols, flavanoids]
    ])

    # Scale the input
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    # Get predicted class
    wine_type = prediction[0]

    st.success(f"Predicted Wine Type: Type-{wine_type}")