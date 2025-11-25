import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model & scaler
model = joblib.load("lopo.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude",
    "RoomsPerPerson", "BedsPerRoom", "PopulationPerHouse"
]

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="California House Price Prediction", layout="wide")

# ========== HEADER ==========
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom:0;'>🏡 California House Price Prediction</h1>
    <p style='text-align:center; font-size:18px; margin-top:0;'>Clean • Modern • Smart</p>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ========== TABS ==========
tab1, tab2 = st.tabs(["🔮 Predict Price", "📊 Visualizations"])

# ============================================================
# ======================= TAB 1: PREDICT =====================
# ============================================================

with tab1:
    st.markdown("### 🔢 Enter House Features")

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

    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population,
                            AveOccup, Latitude, Longitude, RoomsPerPerson,
                            BedsPerRoom, PopulationPerHouse]])

    st.write("")

    # ------- Predict Button -------
    if st.button("🚀 Predict House Price", use_container_width=True):
        processed = scaler.transform(input_data)
        prediction = model.predict(processed)[0]
        price_usd = prediction * 100000

        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:12px;
                background:#F6F6F6;
                text-align:center;
                border:1px solid #DDD;
                margin-top:25px;
            ">
                <h2 style="color:#2C7BE5;">💰 Predicted Price: ${price_usd:,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# ===================== TAB 2: VISUALS =======================
# ============================================================

with tab2:

    st.markdown("### 📊 Model Visualizations")

    vis1, vis2 = st.columns(2)

    # ---------------- Feature Importance -----------------
    with vis1:
        st.subheader("📈 Feature Importance (Small)")
        try:
            importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(4, 3))
            sorted_idx = np.argsort(importances)
            ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
            ax.set_title("Feature Importance", fontsize=10)
            ax.tick_params(axis='both', labelsize=6)
            st.pyplot(fig)
        except:
            st.info("Feature importance not available for this model.")

    # ---------------- Correlation Heatmap ----------------
    with vis2:
        st.subheader("📉 Correlation Heatmap (Small)")
        df_sample = pd.DataFrame(np.random.randn(200, len(feature_names)), columns=feature_names)

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        cax = ax2.matshow(df_sample.corr())
        fig2.colorbar(cax, shrink=0.5)
        ax2.set_title("Correlation Heatmap", fontsize=10)
        st.pyplot(fig2)

    st.write("---")

    # ---------------- Map (Smaller) ----------------
    st.subheader("🗺 House Location Map (Compact)")
    map_data = pd.DataFrame({"lat": [Latitude], "lon": [Longitude]})

    st.map(map_data)

    st.caption("📍 Simplified map view for cleaner UI.")
