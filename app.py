import streamlit as st
import joblib
import numpy as np

# --- Load the pre-trained scaler and model ---
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("❌ Error: 'scaler.pkl' or 'model.pkl' not found. Please ensure they are in the same directory as this script.")
    st.stop()  # Stop the app if files are not found

# --- App Configuration ---
st.set_page_config(
    page_title="Interactive Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# --- Title and Description ---
st.title("📊 Customer Churn Prediction App")
st.write("Welcome to the **Interactive Customer Churn Prediction** application! "
         "Enter customer details below to predict if they are likely to churn.")

st.markdown("---")

# --- Sidebar Info ---
st.sidebar.header("📘 About This App")
st.sidebar.info(
    "This application uses a machine learning model to predict customer churn "
    "based on Age, Gender, Tenure, and Monthly Charges. Just input the values and hit Predict!"
)
st.sidebar.header("🧾 Definitions")
st.sidebar.markdown(
    "- **Gender**: 🚺 Female (1) / 🚹 Male (0)\n"
    "- **Tenure**: Number of months the customer has stayed.\n"
    "- **Monthly Charges**: Average amount charged per month."
)
st.sidebar.markdown("---")

# --- Input Section ---
st.header("📝 Enter Customer Details")

age = st.slider("👴 **Age**", min_value=18, max_value=90, value=35, help="Enter customer's age")
gender = st.radio("🚻 **Gender**", ["Male", "Female"], help="Select the customer's gender")
tenure = st.slider("📆 **Tenure (Months)**", min_value=0, max_value=72, value=24, help="Customer's tenure in months")
monthly_charge = st.slider("💰 **Monthly Charge ($)**", min_value=10.0, max_value=200.0, value=75.0, step=0.5, help="Monthly billing amount")

st.markdown("---")

# --- Prediction Button ---
if st.button("🚀 **Predict Churn**", help="Click to get prediction"):
    gender_val = 1 if gender == "Female" else 0
    input_data = np.array([[age, gender_val, tenure, monthly_charge]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("🚨 **Prediction: This customer is LIKELY to CHURN!** Consider retention strategies.")
        st.balloons()
    else:
        st.success("✅ **Prediction: This customer is UNLIKELY to CHURN.**")
        st.snow()

    st.markdown("---")
    st.info("Prediction based on current input. Adjust values to explore different outcomes.")
else:
    st.info("👈 Fill in the customer details and click **Predict Churn** to see results.")

st.markdown("---")
st.caption("🛠 Developed with ❤️ using Streamlit by Rohit Kumar")
