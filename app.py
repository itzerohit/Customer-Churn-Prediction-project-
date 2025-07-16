# Gender -> 1 Female 0 Male
# Churn -> 1 Yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl'
# order of x is going to be -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'


import streamlit as st
import joblib
import numpy as np

# --- Load the pre-trained scaler and model ---
# Make sure 'scaler.pkl' and 'model.pkl' are in the same directory as your app.py
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Error: 'scaler.pkl' or 'model.pkl' not found. Please ensure they are in the same directory as this script.")
    st.stop() # Stop the app if files are not found

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

# Input fields
age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

# Only one instance of the gender selectbox
gender = st.selectbox("Enter the Gender", ["Male","Female"])

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthlycharge = st.number_input("Enter Monthly Charge", min_value=30.0, max_value=150.0, value=50.0) # Added default value for clarity

st.divider()

predictbutton = st.button("Predict!")

if predictbutton:
    # Convert gender to numerical representation
    gender_selected = 1 if gender == "Female" else 0

    # Create the input array for prediction
    # Ensure the order matches your training data: 'Age', 'Gender', 'Tenure', 'MonthlyCharges'
    X = [age, gender_selected, tenure, monthlycharge]

    # Convert to a NumPy array
    X_array = np.array(X).reshape(1, -1) # Reshape for single sample prediction

    # Scale the input features
    X_scaled = scaler.transform(X_array)

    # Make the prediction
    prediction = model.predict(X_scaled)[0] # Get the first element of the prediction array

    # Interpret the prediction
    predicted = "Yes" if prediction == 1 else "No"

    st.balloons()
    st.success(f"Predicted Churn: **{predicted}**") # Use st.success for better visibility

else:
    st.info("Please enter the values and use the predict button to see the churn prediction.") # Use st.info for better visibility
