import streamlit as st
import pandas as pd
import joblib

# Load trained model and expected feature order
model = joblib.load("opportunity_model.pkl")
expected_features = joblib.load("model_features.pkl")  # Load expected column order

st.title("SaaS Opportunity Prediction")
st.write("Fill the customer details and get the predicted opportunity score.")

# Input column order (excluding encoded categorical variables)
base_features = [
    'Support Ticket Count', 'Support Ticket Decline', 'Jira Ticket Count', 
    'Monthly Consumption', 'Count of enabled warehouses', 'Count of enabled stores', 
    'Customer Tenure (months)', 'Last Engagement Score'
]

# UI input collection
user_input = {}
for col in base_features:
    user_input[col] = st.number_input(f"{col}:", value=0.0)

# Industry and Region dropdowns
industry = st.selectbox('Industry', ['Healthcare', 'Logistics', 'E-commerce', 'Finance', 'Retail'])
region = st.selectbox('Region', ['North', 'South', 'East', 'West', 'Central'])

# Manually build dummies for categorical inputs
for cat in ['Healthcare', 'Logistics', 'E-commerce', 'Finance', 'Retail']:
    user_input[f'Industry_{cat}'] = 1 if industry == cat else 0
for cat in ['North', 'South', 'East', 'West', 'Central']:
    user_input[f'Region_{cat}'] = 1 if region == cat else 0

# Predict button
if st.button("Predict Opportunity"):
    data = pd.DataFrame([user_input])
    data = data.reindex(columns=expected_features)  # Align column order
    prediction = model.predict(data)[0]
    opp_map = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"Predicted Opportunity: **{opp_map[prediction]}**")