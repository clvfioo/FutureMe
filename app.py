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
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("futureme_model.pkl")
encoders = joblib.load("futureme_encoders.pkl")
target_encoder = joblib.load("futureme_target_encoder.pkl")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 52px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 22px;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 20px;
    }

    .description {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .feature-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 180px;
    }

    .feature-title {
        font-size: 20px;
        font-weight: 600;
    }

    .feature-text {
        font-size: 15px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LANDING PAGE
# ============================================================

st.markdown(
    '<div class="main-title">🌱 FutureMe</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Understand your wellbeing. '
    'Understand your future.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="description">
    FutureMe uses machine learning to analyze lifestyle and
    sleep-related information, explain the factors influencing
    its prediction, and provide educational health-awareness
    feedback.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="feature-card">
        <div class="feature-title">🤖 AI Prediction</div>
        <br>
        <div class="feature-text">
        A machine-learning model analyzes the information
        entered and identifies patterns associated with
        sleep-related categories.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">
        <div class="feature-title">🔎 Explainable AI</div>
        <br>
        <div class="feature-text">
        SHAP helps show which factors had the greatest
        influence on the model's prediction.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">
        <div class="feature-title">💡 Awareness</div>
        <br>
        <div class="feature-text">
        FutureMe provides educational awareness suggestions
        based on the information entered.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">⚙️ How FutureMe Works</div>',
    unsafe_allow_html=True
)

st.write(
    """
    **1️⃣ Enter your information**  
    Provide information about sleep, activity, stress,
    and other lifestyle factors.

    **2️⃣ FutureMe processes the information**  
    The data is converted into the format required by the
    trained machine-learning model.

    **3️⃣ AI analyzes the patterns**  
    The trained model compares the information with patterns
    learned from the dataset.

    **4️⃣ FutureMe generates a prediction**  
    The model produces a predicted sleep-related category.

    **5️⃣ Explainable AI explains the result**  
    SHAP identifies the features that had the greatest
    influence on the prediction.

    **6️⃣ FutureMe provides awareness feedback**  
    The system highlights areas that may deserve attention.
    """
)


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📝 Enter Your Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the following information to generate your "
    "FutureMe analysis."
)


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

if st.button(
    "🔍 Analyze with FutureMe",
    use_container_width=True
):

    # ========================================================
    # CREATE USER DATA
    # ========================================================

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


    # ========================================================
    # ENCODE CATEGORICAL VARIABLES
    # ========================================================

    for col in [
        "Gender",
        "Occupation",
        "BMI Category"
    ]:

        user_data[col] = encoders[col].transform(
            user_data[col]
        )


    # ========================================================
    # EXACT TRAINING FEATURE ORDER
    # ========================================================

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
    # RISK LEVEL
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


    # ========================================================
    # RESULT DASHBOARD
    # ========================================================

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "😴 Sleep Prediction",
            predicted_disorder
        )


    with result_col2:

        st.metric(
            "🌱 Wellbeing Score",
            f"{wellbeing_risk:.2f}"
        )


    with result_col3:

        st.metric(
            "📊 Risk Level",
            risk_level
        )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    if risk_level == "Low":

        st.success(
            "🌱 FutureMe's indicators suggest a lower "
            "level of wellbeing risk based on the values entered."
        )

    elif risk_level == "Moderate":

        st.warning(
            "⚠️ FutureMe identified some areas that may "
            "benefit from greater attention."
        )

    else:

        st.warning(
            "⚠️ FutureMe identified several areas that may "
            "benefit from attention."
        )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.divider()

    st.header(
        "🔎 Why did FutureMe make this prediction?"
    )

    st.write(
        "FutureMe uses patterns learned from its training "
        "data to make a prediction. Explainable AI helps "
        "show which inputs had the greatest influence on "
        "the model's result."
    )


    try:

        explainer = shap.TreeExplainer(model)

        shap_result = explainer(user_data)

        shap_values = shap_result.values


        # ----------------------------------------------------
        # HANDLE SHAP OUTPUT
        # ----------------------------------------------------

        if len(shap_values.shape) == 3:

            values = shap_values[
                0,
                :,
                prediction[0]
            ]

        elif len(shap_values.shape) == 2:

            values = shap_values[0]

        else:

            values = shap_values


        # ----------------------------------------------------
        # EXPLANATION TABLE
        # ----------------------------------------------------

        explanation = pd.DataFrame({
            "Feature": user_data.columns,
            "Impact": values
        })


        explanation["Absolute Impact"] = (
            explanation["Impact"].abs()
        )


        explanation = explanation.sort_values(
            "Absolute Impact",
            ascending=False
        )


        top_features = explanation.head(5)


        # ----------------------------------------------------
        # TOP FACTORS
        # ----------------------------------------------------

        st.subheader(
            "Top factors influencing the prediction"
        )


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
        # FEATURE CHART
        # ----------------------------------------------------

        st.subheader(
            "📈 Feature Influence"
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
            "This chart explains how the model used the "
            "provided information. It does not indicate "
            "medical causation."
        )


    except Exception as e:

        st.warning(
            "The prediction was successful, but the "
            "Explainable AI section could not be generated."
        )

        st.write(
            "SHAP error:"
        )

        st.code(
            str(e)
        )


    # ========================================================
    # PERSONALIZED HEALTH AWARENESS
    # ========================================================

    st.divider()

    st.header(
        "💡 Personalized Health Awareness"
    )


    suggestions = []


    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    if sleep_duration < 7:

        suggestions.append(
            "😴 **Sleep:** Your entered sleep duration is below "
            "7 hours. Maintaining a consistent sleep schedule "
            "may support better sleep habits."
        )


    # --------------------------------------------------------
    # SLEEP QUALITY
    # --------------------------------------------------------

    if sleep_quality <= 5:

        suggestions.append(
            "🌙 **Sleep Quality:** Your entered sleep quality "
            "is relatively low. Consider looking at factors "
            "that may be affecting your sleep routine."
        )


    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    if stress >= 7:

        suggestions.append(
            "🧠 **Stress:** Your entered stress level is "
            "relatively high. Healthy stress-management "
            "strategies such as taking breaks, relaxation "
            "activities, or talking to someone you trust "
            "may be helpful."
        )


    # --------------------------------------------------------
    # PHYSICAL ACTIVITY
    # --------------------------------------------------------

    if daily_steps < 8000:

        suggestions.append(
            "🚶 **Physical Activity:** Your entered daily "
            "steps are below 8,000. Adding movement "
            "throughout the day can support an active lifestyle."
        )


    # --------------------------------------------------------
    # HEART RATE
    # --------------------------------------------------------

    if heart_rate > 100:

        suggestions.append(
            "❤️ **Heart Rate:** Your entered heart rate is "
            "relatively high. If this is unusual for you, "
            "consider discussing it with a qualified "
            "healthcare professional."
        )


    # ========================================================
    # DISPLAY PERSONALIZED FEEDBACK
    # ========================================================

    if suggestions:

        for suggestion in suggestions:

            st.write(
                suggestion
            )

    else:

        st.success(
            "🌱 Based on the information entered, FutureMe "
            "did not identify any of its basic awareness flags."
        )


    # ========================================================
    # RISK SCORE BREAKDOWN
    # ========================================================

    st.divider()

    st.header(
        "📋 Risk Score Breakdown"
    )


    risk_col1, risk_col2 = st.columns(2)


    with risk_col1:

        st.metric(
            "🧠 Mental Risk Indicator",
            f"{mental_risk:.2f}"
        )


    with risk_col2:

        st.metric(
            "🏃 Physical Risk Indicator",
            f"{physical_risk:.2f}"
        )


    # ========================================================
    # RISK VISUALIZATION
    # ========================================================

    st.subheader(
        "🌱 Wellbeing Risk Visualization"
    )


    risk_percentage = min(
        max(wellbeing_risk / 10, 0),
        1
    )


    st.progress(
        risk_percentage,
        text=f"Wellbeing Risk: {wellbeing_risk:.2f} / 10"
    )


    st.write(
        "The Mental Risk and Physical Risk values are "
        "custom indicators created for this educational "
        "prototype. They are not medically validated scores."
    )


    # ========================================================
    # HOW FUTUREME WORKS
    # ========================================================

    st.divider()

    st.header(
        "⚙️ How FutureMe Works"
    )


    st.write(
        """
        **1️⃣ User Input**

        The user enters information about sleep, activity,
        stress, and other lifestyle factors.

        **2️⃣ Data Processing**

        FutureMe converts the information into a format
        that the machine-learning model can understand.

        **3️⃣ Machine Learning**

        The trained model analyzes patterns in the data.

        **4️⃣ Prediction**

        FutureMe produces a predicted sleep-related category.

        **5️⃣ Explainable AI**

        SHAP helps identify which input features had the
        greatest influence on the model's prediction.

        **6️⃣ Health Awareness**

        FutureMe provides educational awareness indicators
        based on the information entered.
        """
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ FutureMe is an educational health-awareness "
        "project and is not a medical diagnostic system. "
        "Predictions and risk indicators should not be "
        "used to make medical decisions."
    )
