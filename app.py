import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("SVC.pkl", "rb"))

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn risk.")


st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", 0, 100, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No internet service", "No", "Yes"])
online_backup = st.sidebar.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device_protection = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No internet service", "No", "Yes"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)


data = pd.DataFrame({
    "gender": [0 if gender == "Male" else 1],
    "SeniorCitizen": [senior_citizen],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    "tenure": [tenure],
    "PhoneService": [1 if phone_service == "Yes" else 0],
    "PaperlessBilling": [1 if paperless_billing == "Yes" else 0],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
})

multi_input = {
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaymentMethod": payment_method,
}

multi_df = pd.DataFrame([multi_input])
multi_df = pd.get_dummies(multi_df)

# Merge
X_input = pd.concat([data, multi_df], axis=1)

# -----------------------------
# Feature Engineering
# -----------------------------
X_input["Tenure_to_TotalCharges_Ratio"] = total_charges / (tenure if tenure > 0 else 1)
X_input["High_MonthlyCharges_Flag"] = int(monthly_charges > data["MonthlyCharges"].quantile(0.75))
X_input["Partner_Dependents_Combo"] = int((partner == "Yes") and (dependents == "Yes"))
X_input["PaperlessBilling_Risk"] = 1 if paperless_billing == "Yes" else 0
X_input["ElectronicCheck_Churn_Risk"] = 1 if payment_method == "Electronic check" else 0


X_input = X_input.reindex(columns=model.feature_names_in_, fill_value=0)


if st.sidebar.button("🔮 Predict Churn"):
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"⚠️ The customer is likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ The customer is not likely to churn (Probability: {prob:.2f})")

    st.write("### Input Features Used:")
    st.dataframe(X_input)
