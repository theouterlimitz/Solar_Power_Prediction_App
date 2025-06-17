import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===================================================================
# --- 1. Load the Saved Model and Scaler ---
# ===================================================================

# Use a try-except block to handle potential file errors
try:
    # Load the trained XGBoost model
    model = joblib.load('xgb_model.joblib')
    
    # Load the scaler
    scaler = joblib.load('scaler.joblib')
    
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'xgb_model.joblib' and 'scaler.joblib' are in the same directory as this app.")
    st.stop() # Stop the app if files are not found

# ===================================================================
# --- 2. Create the User Interface (UI) with Streamlit ---
# ===================================================================

# Set the title and a descriptive subheader for the app
st.title('☀️ Solar Power Generation Prediction')
st.markdown("""
This app predicts the **AC Power Output (in kW)** of a solar power plant based on weather conditions and time of day. 
Adjust the sliders on the left to see the model's prediction.
""")

# Create a sidebar for user inputs
st.sidebar.header('Input Features')

# Define a function to get user input using sliders
def user_input_features():
    irradiation = st.sidebar.slider('Irradiation (Sunlight Level)', 0.0, 1.2, 0.5)
    module_temp = st.sidebar.slider('Module Temperature (°C)', 15.0, 75.0, 45.0)
    ambient_temp = st.sidebar.slider('Ambient Temperature (°C)', 10.0, 50.0, 30.0)
    hour = st.sidebar.slider('Hour of Day (24-hour format)', 0, 23, 12)
    minute = st.sidebar.slider('Minute of Hour', 0, 45, 30, step=15)
    month = st.sidebar.slider('Month', 1, 12, 6)
    day_of_year = st.sidebar.slider('Day of Year', 1, 365, 150)
    
    # Create a dictionary of the inputs
    data = {
        'AMBIENT_TEMPERATURE': ambient_temp,
        'MODULE_TEMPERATURE': module_temp,
        'IRRADIATION': irradiation,
        'MONTH': month,
        'DAY_OF_YEAR': day_of_year,
        'HOUR': hour,
        'MINUTE': minute
    }
    
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()


# ===================================================================
# --- 3. Model Prediction and Output Display ---
# ===================================================================

# Display the user's selected input features
st.subheader('Your Selected Input Features:')
st.write(input_df)

# Scale the user's input using the loaded scaler
input_scaled = scaler.transform(input_df)

# Make a prediction using the loaded model
prediction = model.predict(input_scaled)

# Ensure the prediction is not negative
prediction_value = max(0, prediction[0])

# Display the prediction in a prominent way
st.subheader('Predicted AC Power Output:')
# Use st.metric for a nice visual display
st.metric(label="Power Output", value=f"{prediction_value:.2f} kW")

st.markdown("---")
st.write("This app uses an XGBoost Regressor model (R² ≈ 0.96) trained on real-world solar farm data.")
