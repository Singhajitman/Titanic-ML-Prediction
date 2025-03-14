import streamlit as st
import joblib

# **Load model & scaler**
clf = joblib.load("titanic_survival_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚢 Titanic Survival Predictor")

# **Get user input**
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10)
parch = st.number_input("Parents/Children", min_value=0, max_value=10)
fare = st.number_input("Fare Paid ($)", min_value=0.0)
age = st.number_input("Age", min_value=1.0, max_value=100.0)

# **Make prediction**
if st.button("Predict Survival"):
    input_data = [[pclass, sibsp, parch, fare, age]]
    prediction = clf.predict(scaler.transform(input_data))  # Scale input & predict
    st.write("🟢 Survived!" if prediction[0] == 1 else "🔴 Did not survive.")
