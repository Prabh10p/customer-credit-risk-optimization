import streamlit as st
import pandas as pd
from src.Pipeline.Model_pipeline import Pipeline, ModelFeatures

# Streamlit Page Config
st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("🏦 Loan Default Prediction App")
st.markdown("Enter borrower details below to predict whether the loan will default.")

# Input Fields
person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.number_input("Employment Duration (years)", min_value=0.0, max_value=50.0, value=2.0)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input(" Loan Amount ($)", min_value=0.0, value=5000.0)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0)
loan_status = st.selectbox("Loan Status", [0, 1], help="0 = Current, 1 = Default")
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.1)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0.0, value=3.0)

# Predict Button
if st.button("Predict Loan Default"):
    try:
        # Match the exact model feature structure
        features = ModelFeatures(
            person_age,
            person_income,
            person_home_ownership,
            person_emp_length,
            loan_intent,
            loan_grade,
            loan_amnt,
            loan_int_rate,
            loan_status,
            loan_percent_income,
            cb_person_cred_hist_length
        )

        input_df = features.to_dataframe()

        # Optional: Show input for verification
        st.subheader("Input Summary")
        st.write(input_df)

        # Run prediction
        pipeline = Pipeline()
        prediction = pipeline.MakePipeline(input_df)

        # Show result
        st.success(f" **Prediction:** The loan is predicted to be **{'Default' if prediction[0] == 1 else 'Current'}**.")

    except Exception as e:
        st.error(f" An error occurred: {str(e)}")

