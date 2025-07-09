import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration (can be here) ---
MODEL_PATH = 'models/RFR_model.pkl'
FEATURE_NAMES = ['temperature', 'humidity', 'wind_speed', 'general_diffuse_flows', 'diffuse_flows', 'air_quality_index_(pm)']
TARGET_NAME = 'power_consumption_in_a_zone'

st.set_page_config(page_title="Power Consumption Predictor", layout="centered")

@st.cache_resource
def load_ml_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_ml_model(MODEL_PATH)

st.title("Zone Power Consumption Prediction")
st.markdown("---")

st.write("Enter the current environmental and flow data to predict the power consumption in the zone.")

st.sidebar.header("Input Parameters")

def user_input_features():
    temperature = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 25.0, help="Current ambient temperature in Celsius.")
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 60.0, help="Relative humidity percentage.")
    wind_speed = st.sidebar.slider('Wind Speed (km/h)', 0.0, 50.0, 10.0, help="Wind speed in kilometers per hour.")
    general_diffuse_flows = st.sidebar.slider('General Diffuse Flows (W/m²)', 0.0, 2000.0, 500.0, help="Total solar radiation on a horizontal surface in Watts per square meter.")
    diffuse_flows = st.sidebar.slider('Diffuse Flows (W/m²)', 0.0, 1000.0, 200.0, help="Diffuse solar radiation in Watts per square meter.")
    air_quality_index_pm = st.sidebar.slider('Air Quality Index (PM)', 0.0, 500.0, 50.0, help="Particulate Matter (PM) index indicating air quality.")

    data = {
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'general_diffuse_flows': general_diffuse_flows,
        'diffuse_flows': diffuse_flows,
        'air_quality_index_(pm)': air_quality_index_pm
    }
    features_df = pd.DataFrame(data, index=[0]) # Model expects a DataFrame
    return features_df

input_df = user_input_features()

st.subheader('Given Input Parameters:')
st.dataframe(input_df, hide_index=True)

st.markdown("---")

if st.button('Predict Power Consumption'):
    try:
        input_for_prediction = input_df[FEATURE_NAMES] 
        prediction = model.predict(input_for_prediction)

        st.success(f"**Predicted Power Consumption:** {prediction[0]:,.2f} units (e.g., kWh)")
        st.info("*(Units depend on your training data's target variable scale)*")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values and the model file.")

st.markdown("---")
st.caption("This app is for demonstration purposes. Model accuracy depends on training data.")