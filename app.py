import streamlit as st
import pandas as pd
import joblib

# Load trained FutureMe files
model = joblib.load("futureme_model.pkl")
encoders = joblib.load("futureme_encoders.pkl")
target_encoder = joblib.load("futureme_target_encoder.pkl")

st.title("🌱 FutureMe")
st.write("AI-powered health awareness and wellbeing prediction")

st.info(
    "FutureMe is an educational health-awareness prototype. "
    "It does not diagnose medical conditions."
)

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

occupation = st.selectbox(
    "Occupation",
    encoders["Occupation"].classes_.tolist()
)

sleep_duration = st.number_input(
    "Sleep Duration (hours)",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

sleep_quality = st.slider(
    "Quality of Sleep",
    min_value=1,
    max_value=10,
    value=7
)

physical_activity = st.number_input(
    "Physical Activity Level",
    min_value=0,
    value=50
)

stress = st.slider(
    "Stress Level",
    min_value=1,
    max_value=10,
    value=5
)

bmi = st.selectbox(
    "BMI Category",
    encoders["BMI Category"].classes_.tolist()
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=30,
    max_value=200,
    value=75
)

daily_steps = st.number_input(
    "Daily Steps",
    min_value=0,
    value=7000
)

if st.button("🔍 Analyze with FutureMe"):

    user_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": sleep_quality,
        "Physical Activity Level": physical_activity,
        "Stress Level": stress,
        "BMI Category": bmi,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps
    }])

    # Encode categorical variables
    for col in ["Gender", "Occupation", "BMI Category"]:
        user_data[col] = encoders[col].transform(user_data[col])

    # Make sure feature order matches training
    user_data = user_data[
        [
            "Gender",
            "Age",
            "Occupation",
            "Sleep Duration",
            "Quality of Sleep",
            "Physical Activity Level",
            "Stress Level",
            "BMI Category",
            "Heart Rate",
            "Daily Steps"
        ]
    ]

    # Machine-learning prediction
    prediction = model.predict(user_data)

    predicted_disorder = target_encoder.inverse_transform(
        prediction
    )[0]

    # Mental risk indicator
    mental_risk = stress + (10 - sleep_quality)

    # Physical risk indicator
    bmi_score = {
        "Normal": 1,
        "Normal Weight": 1,
        "Overweight": 2,
        "Obese": 3
    }.get(bmi, 1)

    physical_risk = (
        bmi_score +
        (10000 - daily_steps) / 5000
    )

    # Combined wellbeing score
    wellbeing_risk = (
        mental_risk * 0.5 +
        physical_risk * 0.5
    )

    if wellbeing_risk < 4:
        risk_level = "Low"
    elif wellbeing_risk < 7:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    st.divider()

    st.header("📊 FutureMe Results")

    st.subheader("Sleep Prediction")
    st.write(f"**Predicted category:** {predicted_disorder}")

    st.subheader("Wellbeing Risk")
    st.metric(
        "Wellbeing Risk Score",
        f"{wellbeing_risk:.2f}"
    )

    st.write(f"**Risk Level:** {risk_level}")

    st.subheader("💡 Health Awareness")

    if sleep_duration < 7:
        st.write("• Your entered sleep duration is below 7 hours.")

    if stress >= 7:
        st.write("• Your entered stress level is relatively high.")

    if daily_steps < 8000:
        st.write("• Your entered daily activity level could be increased.")

    if sleep_quality <= 5:
        st.write("• Your entered sleep quality is relatively low.")

    st.caption(
        "FutureMe provides educational health-awareness insights "
        "and is not a medical diagnostic tool."
    )
