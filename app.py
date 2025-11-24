import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("lopo.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏡 California House Price Prediction App")
st.write("Enter the house features to predict the price.")

# Input fields
MedInc = st.number_input("Median Income", min_value=0.0, max_value=20.0, value=5.0)
HouseAge = st.number_input("House Age", min_value=1, max_value=60, value=20)
AveRooms = st.number_input("Average Rooms", min_value=0.5, max_value=15.0, value=6.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.0)
Population = st.number_input("Population", min_value=1, max_value=50000, value=1200)
AveOccup = st.number_input("Average Occupancy", min_value=0.5, max_value=10.0, value=3.5)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.19)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.45)

# Create input array
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Predict button
if st.button("Predict House Price"):
    processed = scaler.transform(input_data)
    prediction = model.predict(processed)[0]
    price_usd = prediction * 100000  # convert to real dollars

    st.success(f"Predicted House Price: **${price_usd:,.2f}**")
