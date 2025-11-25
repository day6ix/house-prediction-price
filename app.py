import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model + scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler1.pkl")

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏡",
    layout="centered",
)

# ========= HEADER UI ========= #
st.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <h1 style="color:#2E86C1;">🏡 California House Price Predictor</h1>
        <p style="font-size:17px; color:#555;">
            Enter house details → View prediction → Explore insights
        </p>
    </div>
""", unsafe_allow_html=True)

# ========= TABS ========= #
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Visualizations"])


# =========================================================
#                    TAB 1 — PREDICTION
# =========================================================
with tab1:

    st.markdown("### ✨ Enter House Features")

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        MedInc = st.number_input("Median Income", 0.0, 20.0, 5.0)
        HouseAge = st.number_input("House Age", 1, 60, 20)
        AveRooms = st.number_input("Avg Rooms", 0.5, 15.0, 6.0)
        Population = st.number_input("Population", 1, 50000, 1200)
        RoomsPerPerson = st.number_input("Rooms per Person", 0.001, 1.0, 0.005)

    with col2:
        AveBedrms = st.number_input("Avg Bedrooms", 0.5, 5.0, 1.0)
        AveOccup = st.number_input("Avg Occupancy", 0.5, 10.0, 3.5)
        Latitude = st.number_input("Latitude", 32.0, 42.0, 34.19)
        Longitude = st.number_input("Longitude", -125.0, -114.0, -118.45)
        BedsPerRoom = st.number_input("Beds per Room", 0.05, 1.0, 0.16)

    PopulationPerHouse = st.number_input("Population per House", 1.0, 20000.0, 340.0)

    # Data vector
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population,
                          AveOccup, Latitude, Longitude, RoomsPerPerson,
                          BedsPerRoom, PopulationPerHouse]])

    if st.button("Predict Price 🧮"):
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]
        price = pred * 100000

        st.markdown("""
            <div style="
                background:#EBF5FB;
                padding:20px;
                border-radius:15px;
                text-align:center;
                border:1px solid #AED6F1;">
                <h2 style="color:#1B4F72;">Predicted House Price</h2>
                <h1 style="color:#239B56;">${:,.2f}</h1>
            </div>
        """.format(price), unsafe_allow_html=True)

    # Google map (small)
    st.markdown("### 📍 House Location")
    st.markdown(f"""
        <iframe
            width="100%"
            height="250"
            style="border-radius:15px;"
            loading="lazy"
            src="https://www.google.com/maps?q={Latitude},{Longitude}&hl=en&z=12&output=embed">
        </iframe>
    """, unsafe_allow_html=True)



# =========================================================
#                    TAB 2 — VISUALIZATIONS
# =========================================================
with tab2:

    st.markdown("### 📊 Model Visual Insights")

    # ==== FEATURE IMPORTANCE ==== #
    st.markdown("#### 📈 Feature Importance (small)")

    importances = model.feature_importances_
    feature_names = [
        "MedInc","HouseAge","AveRooms","AveBedrms","Population",
        "AveOccup","Latitude","Longitude","RoomsPerPerson",
        "BedsPerRoom","PopulationPerHouse"
    ]

    fig, ax = plt.subplots(figsize=(5, 3))
    sorted_idx = np.argsort(importances)
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    ax.set_title("Feature Importance", fontsize=11)
    ax.tick_params(labelsize=7)
    st.pyplot(fig)

    st.markdown("---")

    # ==== CORRELATION HEATMAP ==== #
    st.markdown("#### 🔥 Correlation Heatmap (small sample)")

    df = pd.DataFrame(np.random.rand(50, 11), columns=feature_names)

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.heatmap(df.corr(), ax=ax2, cmap="Blues", cbar=False)
    ax2.set_title("Correlation Heatmap", fontsize=11)
    st.pyplot(fig2)
