import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load trained model & encoders
# ===============================
model = joblib.load("XGB_Credit_model.pkl")

encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

# ===============================
# App Configuration
# ===============================
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.title("Credit Risk Prediction App")
st.write(
    """
    This application predicts whether a loan applicant is **Creditworthy (GOOD)** 
    or **High Risk (BAD)** using a Machine Learning model.
    """
)

st.markdown("---")

# ===============================
# User Inputs
# ===============================
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1)

housing = st.selectbox("Housing", ["own", "rent", "free"])
savings_accounts = st.selectbox(
    "Saving accounts", ["little", "moderate", "quite rich", "rich"]
)
checking_accounts = st.selectbox(
    "Checking account", ["little", "moderate", "rich"]
)

credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# ===============================
# Risk Threshold (IMPORTANT)
# ===============================
st.markdown("### Risk Sensitivity")
threshold = st.slider(
    "Probability threshold for approving credit",
    min_value=0.30,
    max_value=0.80,
    value=0.50,
    step=0.05,
)

st.caption(
    "Lower threshold = stricter approval (safer for banks) | "
    "Higher threshold = more approvals (higher risk)"
)

# ===============================
# Prepare Input Data
# ===============================
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([savings_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_accounts])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# ===============================
# Prediction
# ===============================
if st.button("Predict Credit Risk"):
    prob = model.predict_proba(input_df)[0][1]  # Probability of GOOD credit
    prediction = "GOOD" if prob >= threshold else "BAD"

    st.markdown("---")

    if prediction == "GOOD":
        st.success("✅ Credit Risk Assessment: GOOD")
    else:
        st.error("❌ Credit Risk Assessment: BAD")

    st.info(f"**Model Confidence (GOOD Credit Probability):** {prob:.2%}")

    # Business Explanation
    st.markdown("### Decision Explanation")
    st.write(
        f"""
        - The model estimates a **{prob:.2%}** probability that this applicant is creditworthy.
        - Based on the selected threshold (**{threshold:.2f}**), the applicant is classified as **{prediction}**.
        - This probability-based decision aligns with banking risk assessment practices.
        """
    )
