import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Must be FIRST Streamlit call
st.set_page_config(layout="wide")

# ==== Load Models (replace with your paths and preprocessing as needed) ====
@st.cache_resource
def load_models():
    try:
        consumption_model = tf.keras.models.load_model("electricity_consumption_model.h5")
    except Exception:
        consumption_model = None
    try:
        bill_model = tf.keras.models.load_model("bill_predictor.h5")
    except Exception:
        bill_model = None
    return consumption_model, bill_model

consumption_model, bill_model = load_models()

# ==== Sidebar Navigation ====
st.sidebar.title("Smart Electricity Dashboard")
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Live Usage",
    "Monthly Bill Forecast",
    "Alerts & Notifications",
    "Settings",
    "AI Assist"
])

st.title("Smart Electricity Monitoring & Forecasting")

# ==== Dashboard Section ====
if page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Consumption", "5.7 kWh")
    col2.metric("Forecasted Monthly Bill", "₹1200")
    col3.metric("Today's Peak Usage", "6.3 kWh")
    col4.info("Seasonal Impact: AC load ↑ (34°C)")

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Real-Time Consumption")
        times = np.arange(10, 19, 1)
        consumption = np.random.uniform(2, 8, len(times))
        fig, ax = plt.subplots()
        ax.plot(times, consumption, marker="o")
        ax.set_xlabel("Hour")
        ax.set_ylabel("kWh")
        ax.set_title("Last 1 hour")
        st.pyplot(fig)

    with col6:
        st.subheader("Monthly Bill Forecast")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
        bills = np.random.randint(900, 2200, size=len(months))
        predicted = bills + np.random.randint(50, 300, size=len(months))
        fig2, ax2 = plt.subplots()
        width = 0.3
        ax2.bar(np.arange(len(months)), bills, width=width, label="Actual")
        ax2.bar(np.arange(len(months))+width, predicted, width=width, label="Predicted")
        ax2.set_xticks(np.arange(len(months))+width/2)
        ax2.set_xticklabels(months)
        ax2.set_ylabel("₹")
        ax2.legend()
        st.pyplot(fig2)

    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Level-wise Usage")
        labels = ["High", "Medium", "Low"]
        values = [30, 45, 25]
        fig3, ax3 = plt.subplots()
        ax3.pie(values, labels=labels, autopct='%1.1f%%', colors=['#fd7e14','#ffe066','#51cf66'])
        st.pyplot(fig3)

    with col8:
        st.subheader("AI Insights")
        st.info("Your usage may result in a higher bill this summer.")

# ==== Live Usage (IoT Data Display Placeholder) ====
if page == "Live Usage":
    st.header("Live Usage")
    st.write("Stream live metrics here from your IoT hardware or backend API.")

# ==== Monthly Bill Forecast ====
if page == "Monthly Bill Forecast":
    st.header("Monthly Bill Forecast")
    st.write("Graphs and AI prediction details for your monthly bill.")

# ==== Alerts & Notifications ====
if page == "Alerts & Notifications":
    st.header("Alerts & Notifications")
    st.warning("No critical alerts at the moment.")

# ==== Settings ====
if page == "Settings":
    st.header("Settings")
    st.write("Add device, set thresholds, and configure notifications.")

# ==== AI Assist ====
if page == "AI Assist":
    st.header("AI Assist: Predict Your Usage and Bill")
    st.caption("Enter all 13 parameters used during model training")

    # User Inputs - Match exactly with model input structure (13 features)
    temperature = st.number_input("Temperature (°C)", value=30.0)
    humidity = st.number_input("Humidity (%)", value=60.0)
    wind_speed = st.number_input("Wind Speed (m/s)", value=2.0)
    load_factor = st.slider("Load Factor", 0.0, 1.0, 0.75)
    month_sin = st.number_input("Month sin", value=0.0)
    month_cos = st.number_input("Month cos", value=1.0)
    pressure = st.number_input("Pressure (Pa)", value=101325.0)
    visibility = st.number_input("Visibility (km)", value=5.0)
    dew_point = st.number_input("Dew Point (°C)", value=20.0)
    solar_radiation = st.number_input("Solar Radiation (W/m²)", value=250.0)
    weekday = st.number_input("Day of Week (0=Mon,6=Sun)", min_value=0, max_value=6, value=1)
    hour_sin = st.number_input("Hour sin (Cyclic Time Encoding)", value=0.0)
    total_kwh = st.number_input("Historical Total kWh", value=300.0)

    # Combine all features into same order model expects
    input_features = np.array([[temperature, humidity, wind_speed, 
                                load_factor, month_sin, month_cos, 
                                pressure, visibility, dew_point, 
                                solar_radiation, weekday, hour_sin, total_kwh]])

    if st.button("Predict"):
        predicted_kwh = None
        predicted_bill = None

        # --- Consumption Prediction ---
        if consumption_model:
            try:
                predicted_kwh = float(consumption_model.predict(input_features)[0][0])
                st.success(f"Predicted Consumption: {predicted_kwh:.2f} kWh")
            except Exception as e:
                st.error(f"Error in consumption model: {e}")
        else:
            st.warning("Consumption model not loaded. Showing demo value.")
            predicted_kwh = 534.62
            st.success(f"Predicted Consumption: {predicted_kwh:.2f} kWh")

        # --- Bill Prediction ---
        if bill_model:
            try:
                predicted_bill = float(bill_model.predict(input_features)[0][0])
                st.success(f"Predicted Bill: ₹{predicted_bill:.2f}")
            except Exception as e:
                st.error(f"Error in bill model: {e}")
        else:
            st.warning("Bill model not loaded. Showing demo value.")
            predicted_bill = 3180.75
            st.success(f"Predicted Bill: ₹{predicted_bill:.2f}")
