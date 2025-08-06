import streamlit as st
import pandas as pd
from model import load_model
from feature_engineering import engineer_features
from utils import predict_score

st.title("Predict CIBIL Score")

st.write("Fill out the form to get CIBIL score.")

monthly_income = st.number_input("Monthly Income", min_value=0)
total_txn_amount = st.number_input("Total Digital Transactions amount in 6 months", min_value=0)
active_months = st.number_input("Number of Active Months", min_value=1, value=6)
loan_count = st.number_input("No of utility bills", min_value=0)
on_time_payments = st.number_input("Number of On-Time Payments", min_value=0)

if st.button("Predict Score"):
    input_data = pd.DataFrame.from_dict({
        'monthly_income': [monthly_income],
        'total_txn_amount': [total_txn_amount],
        'active_months': [active_months],
        'loan_count': [loan_count],
        'on_time_payments': [on_time_payments]
    })

    model = load_model()
    input_data = engineer_features(input_data)
    score = predict_score(model, input_data.iloc[0])
    
    st.success(f"Predicted Shadow Score: {score}")
    
    
    
    st.success(f"Check: {monthly_income}")
    st.success(f"Check: {total_txn_amount}")
    st.success(f"check: {active_months}")
    st.success(f"check: {loan_count}")
    st.success(f"check: {on_time_payments}")
    