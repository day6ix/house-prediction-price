import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("lopo.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names (must match your model order)
feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude",
    "RoomsPerPerson", "BedsPerRoom", "PopulationPerHouse"
]

# ========== PAGE HEADER ==========

st.markdown("<h1 style='text-align:center;'>🏡 California House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Enter the house features to predict the price.</p>", unsafe_allow_html=True)

st.write("---")

# ========== INPUT AREA (CLEAN UI CARDS) ==========

st.markdown("### 🔢 Input House Features")
col1, col2, col3 = st.columns(3)

with col1:
    MedInc = st.number_input("Median Income", 0.0, 20.0, 5.0)
    HouseAge = st.number_input("House Age", 1, 60, 20)
    AveRooms = st.number_input("Average Rooms", 0.5, 15.0, 6.0)
    AveBedrms = st.number_input("Average Bedrooms", 0.5, 5.0, 1.0)

with col2:
    Population = st.number_input("Population", 1, 50000, 1200)
    AveOccup = st.number_input("Average Occupancy", 0.5, 10.0, 3.5)
    Latitude = st.number_input("Latitude", 32.0, 42.0, 34.19)
    Longitude = st.number_input("Longitude", -125.0, -114.0, -118.45)

with col3:
    RoomsPerPerson = st.number_input("RoomsPerPerson", 0.001, 0.1, 0.005)
    BedsPerRoom = st.number_input("BedsPerRoom", 0.05, 0.5, 0.1667)
    PopulationPerHouse = st.number_input("PopulationPerHouse", 1.0, 20000.0, 342.85)

# Prepare input
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population,
                        AveOccup, Latitude, Longitude, RoomsPerPerson,
                        BedsPerRoom, PopulationPerHouse]])

# ========== PREDICTION ==========

if st.button("🔮 Predict House Price", use_container_width=True):
    processed = scaler.transform(input_data)
    prediction = model.predict(processed)[0]
    price_usd = prediction * 100000

    st.success(f"### 💰 Predicted Price: **${price_usd:,.2f}**")

    st.write("---")

    # ===============================================================
    # SMALLER FEATURE IMPORTANCE CHART
    # ===============================================================

    st.subheader("📈 Feature Importance (Smaller)")
    
    try:
        importances = model.feature_importances_
        fig, ax = plt.subplots(figsize=(5, 4))
        sorted_idx = np.argsort(importances)
        ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        ax.set_title("Feature Importance", fontsize=12)
        ax.tick_params(axis='both', labelsize=8)
        st.pyplot(fig)
    except:
        st.info("Feature importance unavailable for this model.")

    st.write("---")

    # ===============================================================
    # SMALLER CORRELATION HEATMAP (SAMPLE)
    # ===============================================================

    st.subheader("📉 Correlation Heatmap (Smaller)")
    
    df_sample = pd.DataFrame(np.random.randn(200, len(feature_names)), columns=feature_names)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    cax = ax2.matshow(df_sample.corr())
    fig2.colorbar(cax, shrink=0.7)
    ax2.set_title("Correlation Heatmap", fontsize=12)

    st.pyplot(fig2)

    st.write("---")

    # ===============================================================
    # SMALLER MAP DISPLAY
    # ===============================================================

    st.subheader("🗺 House Location Map (Smaller)")

    map_data = pd.DataFrame({"lat": [Latitude], "lon": [Longitude]})

    st.map(map_data, size=20)

    st.caption("📍 Location scaled down for cleaner layout.")
