import streamlit as st
import joblib

st.title("🚢 Titanic Survival Predictor")

# **Try to load the model & scaler safely**
try:
    clf = joblib.load("titanic_survival_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("⚠️ Model or scaler file not found! Please check if they are uploaded correctly.")
    st.stop()  # Stop execution if files are missing

# **Get user input**
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10)
parch = st.number_input("Parents/Children", min_value=0, max_value=10)
fare = st.number_input("Fare Paid ($)", min_value=0.0)
age = st.number_input("Age", min_value=1.0, max_value=100.0)

# **Make prediction**
if st.button("Predict Survival"):
    input_data = [[pclass, sibsp, parch, fare, age]]
    try:
        prediction = clf.predict(scaler.transform(input_data))  # ✅ Scale input & predict
        st.write("🟢 Survived!" if prediction[0] == 1 else "🔴 Did not survive.")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
