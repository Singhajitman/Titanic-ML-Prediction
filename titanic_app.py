import streamlit as st
import joblib
import numpy as np

st.title("🚢 Titanic Survival Predictor")

# **Try to load the model & scaler safely**
try:
    clf = joblib.load("titanic_survival_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("⚠️ Model or scaler file not found! Please check if they are uploaded correctly.")
    st.stop()

# **Get user input**
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])  # ✅ New input
sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10)
parch = st.number_input("Parents/Children", min_value=0, max_value=10)
fare = st.number_input("Fare Paid ($)", min_value=0.0)
age = st.number_input("Age", min_value=1.0, max_value=100.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q", "U"])  # ✅ New input

# **Convert "Sex" & "Embarked" to match LabelEncoder**
sex_encoded = 1 if sex == "Male" else 0  # Assuming Male=1, Female=0 from LabelEncoder
embarked_encoded = {"S": 0, "C": 1, "Q": 2, "U": 3}[embarked]  # Map categories to numbers

# **Make prediction**
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, sibsp, parch, fare, age, embarked_encoded]])  # ✅ Now has 7 features
    try:
        input_data_scaled = scaler.transform(input_data)  # Scale input
        prediction = clf.predict(input_data_scaled)  # Predict survival
        st.write("🟢 Survived!" if prediction[0] == 1 else "🔴 Did not survive.")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
