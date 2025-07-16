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

# --- App Configuration ---
st.set_page_config(
    page_title="Interactive Churn Prediction",
    page_icon="📊",
    layout="centered" # Can be "wide" or "centered"
)

# --- Title and Description ---
st.title("📊 Customer Churn Prediction App")
st.write("Welcome to the **Interactive Customer Churn Prediction** application! Enter customer details below to predict if they are likely to churn.")

st.markdown("---")

# --- Sidebar for additional info or settings (optional) ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This application uses a pre-trained machine learning model to predict customer churn based on various demographic and service-related factors. "
    "The model considers **Age**, **Gender**, **Tenure**, and **Monthly Charges** to make its predictions. "
    "Simply input the details and click 'Predict'!"
)
st.sidebar.markdown("---")
st.sidebar.header("Definitions")
st.sidebar.markdown(
    "- **Gender**: 🚺 Female (1) / 🚹 Male (0)\n"
    "- **Tenure**: Number of months the customer has stayed with the company.\n"
    "- **Monthly Charges**: The amount charged to the customer monthly."
)

# --- Main Input Section ---
st.header("Enter Customer Details")

# Input fields with improved labeling and default values
age = st.slider("👴 **Age**", min_value=18, max_value=90, value=35, help="Enter the customer's age.")
gender_option = st.radio("🚻 **Gender**", ["Male", "Female"], help="Select the customer's gender.")
tenure = st.slider("🗓️ **Tenure (Months)**", min_value=0, max_value=72, value=24, help="Enter the number of months the customer has been with the company.")
monthly_charge = st.slider("💸 **Monthly Charge ($)**", min_value=10.0, max_value=200.0, value=75.0, step=0.5, help="Enter the customer's average monthly bill.")

st.markdown("---")

# --- Prediction Button ---
if st.button("🚀 **Predict Churn**", help="Click to get the churn prediction."):
    # Convert gender to numerical representation
    gender_numerical = 1 if gender_option == "Female" else 0

    # Create the input array for prediction
    # Ensure the order matches your training data: 'Age', 'Gender', 'Tenure', 'MonthlyCharges'
    X = [age, gender_numerical, tenure, monthly_charge]

    # Convert to a NumPy array
    X_array = np.array(X).reshape(1, -1) # Reshape for single sample prediction

    # Scale the input features
    X_scaled = scaler.transform(X_array)

    # Make the prediction
    prediction = model.predict(X_scaled)[0] # Get the first element of the prediction array

    # Interpret the prediction
    if prediction == 1:
        st.error("🚨 **Prediction: This customer is LIKELY to CHURN!** Consider retention strategies.")
        st.balloons()
    else:
        st.success("✅ **Prediction: This customer is UNLIKELY to CHURN.**")
        st.snow()

    st.markdown("---")
    st.info("The prediction is based on the provided inputs. Please review the details carefully.")
else:
    st.info("👈 Fill in the customer details and click 'Predict Churn' to see the outcome.")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
