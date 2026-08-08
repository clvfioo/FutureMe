import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="FutureMe",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL AND ENCODERS
# ============================================================

model = joblib.load("futureme_model.pkl")
encoders = joblib.load("futureme_encoders.pkl")
target_encoder = joblib.load("futureme_target_encoder.pkl")


# ============================================================
# TITLE
# ============================================================

st.title("🌱 FutureMe")

st.write(
    "### AI-powered health awareness and wellbeing prediction system"
)

st.info(
    "FutureMe is an educational health-awareness prototype. "
    "It does not diagnose medical conditions."
)


# ============================================================
# USER INPUTS
# ============================================================

st.header("📝 Enter Your Information")


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


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
    value=7.0,
    step=0.1
)


sleep_quality = st.slider(
    "Quality of Sleep",
    min_value=1,
    max_value=10,
    value=7,
    step=1
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
    value=5,
    step=1
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


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze with FutureMe"):

    # --------------------------------------------------------
    # CREATE USER DATAFRAME
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ENCODE CATEGORICAL VARIABLES
    # --------------------------------------------------------

    for col in ["Gender", "Occupation", "BMI Category"]:

        user_data[col] = encoders[col].transform(
            user_data[col]
        )


    # --------------------------------------------------------
    # MAKE SURE FEATURE ORDER MATCHES TRAINING
    # --------------------------------------------------------

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


    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    prediction = model.predict(user_data)

    predicted_disorder = target_encoder.inverse_transform(
        prediction
    )[0]


    # ========================================================
    # MENTAL RISK INDICATOR
    # ========================================================

    mental_risk = (
        stress +
        (10 - sleep_quality)
    )


    # ========================================================
    # PHYSICAL RISK INDICATOR
    # ========================================================

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


    # ========================================================
    # COMBINED WELLBEING SCORE
    # ========================================================

    wellbeing_risk = (
        mental_risk * 0.5 +
        physical_risk * 0.5
    )


    # ========================================================
    # DETERMINE RISK LEVEL
    # ========================================================

    if wellbeing_risk < 4:

        risk_level = "Low"

    elif wellbeing_risk < 7:

        risk_level = "Moderate"

    else:

        risk_level = "High"


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 FutureMe Results")


    # --------------------------------------------------------
    # SLEEP PREDICTION
    # --------------------------------------------------------

    st.subheader("😴 Sleep Prediction")

    st.write(
        f"**Predicted category:** {predicted_disorder}"
    )


    # --------------------------------------------------------
    # WELLBEING SCORE
    # --------------------------------------------------------

    st.subheader("🌱 Wellbeing Risk")

    st.metric(
        "Wellbeing Risk Score",
        f"{wellbeing_risk:.2f}"
    )

    st.write(
        f"**Risk Level:** {risk_level}"
    )


   # ========================================================
# EXPLAINABLE AI
# ========================================================

st.divider()

st.header("🔎 Why did FutureMe make this prediction?")

st.write(
    "FutureMe uses the trained machine-learning model to identify "
    "which factors had the greatest influence on its prediction."
)

try:

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_explanation = explainer(user_data)

    # Get SHAP values
    shap_values = shap_explanation.values

    # SHAP may return:
    # (samples, features)
    # or
    # (samples, features, classes)

    if len(shap_values.shape) == 3:

        # Select the predicted class
        values = shap_values[0, :, prediction[0]]

    elif len(shap_values.shape) == 2:

        values = shap_values[0]

    else:

        values = shap_values


    # Create explanation table
    explanation = pd.DataFrame({
        "Feature": user_data.columns,
        "Impact": values
    })


    # Absolute influence
    explanation["Absolute Impact"] = (
        explanation["Impact"].abs()
    )


    # Sort by importance
    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )


    # Get top 5
    top_features = explanation.head(5)


    st.subheader(
        "Top factors influencing the prediction"
    )


    # Display explanations
    for _, row in top_features.iterrows():

        feature = row["Feature"]
        impact = row["Impact"]

        if impact > 0:

            direction = "increased"

        else:

            direction = "decreased"

        st.write(
            f"**{feature}** — "
            f"{direction} the model's prediction influence."
        )


    # ========================================================
    # FEATURE INFLUENCE CHART
    # ========================================================

    st.subheader("📈 Feature Influence")


    chart_data = top_features.sort_values(
        "Impact"
    )


    fig, ax = plt.subplots()


    ax.barh(
        chart_data["Feature"],
        chart_data["Impact"]
    )


    ax.set_xlabel(
        "Model influence"
    )


    ax.set_ylabel(
        "Feature"
    )


    ax.set_title(
        "Top Factors Influencing Prediction"
    )


    st.pyplot(fig)

    plt.close(fig)


    st.caption(
        "This chart explains how the model used the "
        "provided information. It does not show medical "
        "causes or diagnoses."
    )


except Exception as e:

    st.warning(
        "The prediction was successful, but the "
        "Explainable AI section could not be generated."
    )

    st.write(
        "SHAP error:"
    )

    st.code(str(e))

        # ----------------------------------------------------
        # HANDLE DIFFERENT SHAP OUTPUT FORMATS
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            values = shap_values[prediction[0]]

        else:

            values = shap_values


        # ----------------------------------------------------
        # CREATE EXPLANATION TABLE
        # ----------------------------------------------------

        explanation = pd.DataFrame({
            "Feature": user_data.columns,
            "Impact": values[0]
        })


        explanation["Absolute Impact"] = (
            explanation["Impact"].abs()
        )


        explanation = explanation.sort_values(
            "Absolute Impact",
            ascending=False
        )


        # ----------------------------------------------------
        # TOP 5 FEATURES
        # ----------------------------------------------------

        top_features = explanation.head(5)


        st.write(
            "### Top factors influencing the prediction"
        )


        # ----------------------------------------------------
        # DISPLAY FACTORS
        # ----------------------------------------------------

        for _, row in top_features.iterrows():

            feature = row["Feature"]
            impact = row["Impact"]


            if impact > 0:

                direction = "increased"

            else:

                direction = "decreased"


            st.write(
                f"**{feature}** — "
                f"{direction} the model's prediction influence."
            )


        # ----------------------------------------------------
        # SHAP BAR CHART
        # ----------------------------------------------------

        st.write(
            "### 📈 Feature Influence"
        )


        chart_data = top_features.sort_values(
            "Impact"
        )


        fig, ax = plt.subplots()


        ax.barh(
            chart_data["Feature"],
            chart_data["Impact"]
        )


        ax.set_xlabel(
            "Model influence"
        )


        ax.set_ylabel(
            "Feature"
        )


        ax.set_title(
            "Top Factors Influencing Prediction"
        )


        st.pyplot(fig)

        plt.close(fig)


        st.caption(
            "Higher absolute values indicate a greater influence "
            "on the model's prediction. This explains the model's "
            "behavior and does not indicate medical causation."
        )


    except Exception as e:

        st.warning(
            "The prediction was successful, but the "
            "Explainable AI section could not be generated."
        )

        st.caption(
            f"Technical information: {str(e)}"
        )


    # ========================================================
    # HEALTH AWARENESS
    # ========================================================

    st.divider()

    st.header("💡 Health Awareness")


    suggestions = []


    if sleep_duration < 7:

        suggestions.append(
            "😴 Your entered sleep duration is below 7 hours."
        )


    if stress >= 7:

        suggestions.append(
            "🧠 Your entered stress level is relatively high."
        )


    if daily_steps < 8000:

        suggestions.append(
            "🚶 Your entered daily activity level could be increased."
        )


    if sleep_quality <= 5:

        suggestions.append(
            "🌙 Your entered sleep quality is relatively low."
        )


    if heart_rate > 100:

        suggestions.append(
            "❤️ Your entered heart rate is relatively high. "
            "If this is unusual for you, consider discussing it "
            "with a qualified healthcare professional."
        )


    # --------------------------------------------------------
    # DISPLAY SUGGESTIONS
    # --------------------------------------------------------

    if len(suggestions) > 0:

        for suggestion in suggestions:

            st.write(suggestion)

    else:

        st.success(
            "Your entered values do not trigger any of "
            "FutureMe's basic awareness suggestions."
        )


    # ========================================================
    # RISK SCORE BREAKDOWN
    # ========================================================

    st.divider()

    st.header("📋 Risk Score Breakdown")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Mental Risk Indicator",
            f"{mental_risk:.2f}"
        )


    with col2:

        st.metric(
            "Physical Risk Indicator",
            f"{physical_risk:.2f}"
        )


    st.write(
        "The Mental Risk and Physical Risk values are "
        "custom indicators created for this educational "
        "prototype. They are not medically validated scores."
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ FutureMe is an educational health-awareness "
        "project and is not a medical diagnostic system. "
        "Predictions and risk indicators should not be used "
        "to make medical decisions."
    )
