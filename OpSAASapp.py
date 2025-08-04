import pandas as pd

# Load data (adjust path as needed)
df = pd.read_csv("sample_ai_opportunity_data_with_labels.csv")

# See a sample
print(df.head())


# Drop columns not useful for prediction
drop_cols = ['Account', 'Date']
df = df.drop(columns=drop_cols)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Industry', 'Region'])

# Target encoding (if not already numeric)
target_map = {'Low':0, 'Medium':1, 'High':2}
df['Opportunity Score'] = df['Opportunity Score'].map(target_map)

# Features and target
X = df.drop('Opportunity Score', axis=1)
y = df['Opportunity Score']


from sklearn.model_selection import train_test_split

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Predict
y_pred = model.predict(X_test)

# Evaluation report
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix

# Predict
y_pred = gbc.predict(X_test)

# Evaluate performance
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))




import joblib
joblib.dump(gbc, "opportunity_model.pkl")


import streamlit as st


# Load trained model
model = joblib.load("opportunity_model.pkl")

# Input column order (match model training)
feature_columns = [
    'Support Ticket Count', 'Support Ticket Decline', 'Jira Ticket Count', 
    'Monthly Consumption', 'Consumption % as per committed transactions',
    'POC NPS', 'DM NPS', 'Count of enabled B2C channels', 'Count of enabled B2B channels', 
    'Count of enabled warehouses', 'Count of enabled stores', 'Customer Tenure (months)', 
    'Last Engagement Score',
    # Add your actual dummies for Industry, Region (e.g., 'Industry_Healthcare', etc.)
    # See 'pd.get_dummies' output for your full list
]

st.title("SaaS Opportunity Prediction")
st.write("Fill the customer details and get the predicted opportunity score.")

# Dictionary to gather user input
user_input = {}
for col in feature_columns:
    user_input[col] = st.number_input(f"{col}:", value=0.0)

# If using dummy variables from categorical features, also allow selection:
# (Example)
industry = st.selectbox('Industry', ['Healthcare', 'Logistics', 'E-commerce', 'Finance', 'Retail'])
region = st.selectbox('Region', ['North', 'South', 'East', 'West', 'Central'])

# Manually build dummies
for cat in ['Healthcare', 'Logistics', 'E-commerce', 'Finance', 'Retail']:
    user_input[f'Industry_{cat}'] = 1 if industry == cat else 0
for cat in ['North', 'South', 'East', 'West', 'Central']:
    user_input[f'Region_{cat}'] = 1 if region == cat else 0

if st.button("Predict Opportunity"):
    data = pd.DataFrame([user_input])
    prediction = model.predict(data)[0]
    opp_map = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"Predicted Opportunity: **{opp_map[prediction]}**")