import streamlit as st
import joblib
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🔧",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_data = joblib.load("predictive_maintenance_model.pkl")

model = model_data["model"]
type_mapping = model_data["type_mapping"]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8fafc;
    }

    .main-title {
        text-align: center;
        color: #172033;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-title {
        color: #172033;
        font-size: 25px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
    }

    div.stButton > button {
        background-color: #2563eb;
        color: white;
        font-size: 17px;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔧 About the Model")

    st.write(
        "This application uses a Machine Learning model "
        "to predict whether a machine may experience failure."
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("**Random Forest Classifier**")
    st.write("Number of Trees: **200**")
    st.write("Class Weight: **Balanced**")

    st.divider()

    st.subheader("📊 Prediction Classes")

    st.write("🟢 **0 — No Machine Failure**")
    st.write("🔴 **1 — Machine Failure**")

    st.divider()

    st.subheader("⚙️ Input Features")

    st.write("• Machine Type")
    st.write("• Air Temperature")
    st.write("• Process Temperature")
    st.write("• Rotational Speed")
    st.write("• Torque")
    st.write("• Tool Wear")
    st.write("• Temperature Difference")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔧 Predictive Maintenance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Failure Prediction using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MACHINE PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Machine Parameters</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the current machine operating conditions below."
)


# ============================================================
# INPUT ROW 1
# ============================================================

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["H", "L", "M"]
    )


with col2:

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=200.0,
        max_value=400.0,
        value=300.0,
        step=0.1
    )


# ============================================================
# INPUT ROW 2
# ============================================================

col3, col4 = st.columns(2)

with col3:

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=200.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )


with col4:

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=0,
        max_value=5000,
        value=1500,
        step=10
    )


# ============================================================
# INPUT ROW 3
# ============================================================

col5, col6 = st.columns(2)

with col5:

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )


with col6:

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


# ============================================================
# TEMPERATURE DIFFERENCE
# ============================================================

temp_difference = process_temperature - air_temperature

st.info(
    f"🌡️ **Temperature Difference:** "
    f"{temp_difference:.2f} K"
)


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Machine Failure",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Encode Machine Type
    # --------------------------------------------------------

    type_encoded = type_mapping[machine_type]


    # --------------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            type_encoded,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
            temp_difference
        ]],
        columns=[
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Temp Difference"
        ]
    )


    # --------------------------------------------------------
    # Make Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------------
    # Prediction Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(input_data)[0]

    predicted_index = list(model.classes_).index(prediction)

    confidence = probabilities[predicted_index] * 100


    # ========================================================
    # RESULT SECTION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # FAILURE RESULT
    # ========================================================

    if prediction == 1:

        st.error(
            "🔴 MACHINE FAILURE DETECTED"
        )

        st.write(
            "The model predicts that the machine may "
            "experience a failure. Preventive maintenance "
            "should be considered."
        )

        st.warning(
            "🔧 Recommendation: Inspect the machine and "
            "schedule preventive maintenance."
        )


    # ========================================================
    # NORMAL RESULT
    # ========================================================

    else:

        st.success(
            "🟢 NO MACHINE FAILURE DETECTED"
        )

        st.write(
            "The model predicts that the machine is "
            "operating normally and no failure is "
            "currently expected."
        )

        st.info(
            "👍 Recommendation: Continue normal operation "
            "and monitor machine parameters regularly."
        )


    # ========================================================
    # MODEL CONFIDENCE
    # ========================================================

    st.subheader("🎯 Model Confidence")

    confidence_col1, confidence_col2 = st.columns(2)

    with confidence_col1:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    with confidence_col2:

        if prediction == 1:

            st.metric(
                "Prediction",
                "Failure"
            )

        else:

            st.metric(
                "Prediction",
                "Normal"
            )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader("📋 Input Summary")

    summary = pd.DataFrame(
        {
            "Parameter": [
                "Machine Type",
                "Air Temperature [K]",
                "Process Temperature [K]",
                "Rotational Speed [rpm]",
                "Torque [Nm]",
                "Tool Wear [min]",
                "Temperature Difference [K]"
            ],

            "Value": [
                machine_type,
                f"{air_temperature:.2f}",
                f"{process_temperature:.2f}",
                f"{rotational_speed}",
                f"{torque:.2f}",
                f"{tool_wear}",
                f"{temp_difference:.2f}"
            ]
        }
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

st.write(
    "The following metrics were obtained from the held-out "
    "test dataset containing 2,000 observations."
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(
        "Accuracy",
        "98.70%"
    )

with metric2:

    st.metric(
        "Precision",
        "87.50%"
    )

with metric3:

    st.metric(
        "Recall",
        "72.06%"
    )

with metric4:

    st.metric(
        "F1 Score",
        "79.03%"
    )


# ============================================================
# METRIC EXPLANATION
# ============================================================

st.subheader("📚 Metric Interpretation")

metric_info = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Meaning": [
            "Percentage of all predictions that were correct.",
            "Percentage of predicted failures that were actually failures.",
            "Percentage of actual failures successfully detected.",
            "Balance between precision and recall."
        ],

        "Result": [
            "98.70%",
            "87.50%",
            "72.06%",
            "79.03%"
        ]
    }
)


st.dataframe(
    metric_info,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔍 Confusion Matrix")

confusion_matrix_data = pd.DataFrame(
    [
        [1925, 7],
        [19, 49]
    ],
    index=[
        "Actual: No Failure",
        "Actual: Failure"
    ],
    columns=[
        "Predicted: No Failure",
        "Predicted: Failure"
    ]
)


st.dataframe(
    confusion_matrix_data,
    use_container_width=True
)


# ============================================================
# CONFUSION MATRIX INTERPRETATION
# ============================================================

st.write("### Confusion Matrix Interpretation")

cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)

with cm_col1:

    st.metric(
        "True Negatives",
        "1925"
    )

with cm_col2:

    st.metric(
        "False Positives",
        "7"
    )

with cm_col3:

    st.metric(
        "False Negatives",
        "19"
    )

with cm_col4:

    st.metric(
        "True Positives",
        "49"
    )


# ============================================================
# PREDICTIVE MAINTENANCE INSIGHT
# ============================================================

st.info(
    "💡 **Predictive Maintenance Insight:** "
    "Recall is particularly important because a false negative "
    "means an actual machine failure was not detected. "
    "Therefore, the model should be evaluated using recall "
    "and F1-score in addition to accuracy."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🔍 Feature Importance</div>',
    unsafe_allow_html=True
)

st.write(
    "Feature importance shows how strongly each input "
    "feature contributes to the Random Forest model's decisions."
)


# ============================================================
# FEATURE IMPORTANCE DATA
# ============================================================

feature_names = [
    "Machine Type",
    "Air Temperature [K]",
    "Process Temperature [K]",
    "Rotational Speed [rpm]",
    "Torque [Nm]",
    "Tool Wear [min]",
    "Temperature Difference [K]"
]

importance_values = model.feature_importances_


feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance_values
    }
)


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# FEATURE IMPORTANCE TABLE
# ============================================================

display_table = feature_importance.copy()

display_table["Importance"] = (
    display_table["Importance"] * 100
).round(2)

display_table["Importance"] = (
    display_table["Importance"].astype(str) + "%"
)


st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FEATURE IMPORTANCE BAR CHART
# ============================================================

chart_data = feature_importance.set_index("Feature")

st.bar_chart(
    chart_data["Importance"]
)


# ============================================================
# TOP FEATURES
# ============================================================

top_features = feature_importance.head(3)

st.subheader("🏆 Most Important Features")

for index, row in top_features.iterrows():

    percentage = row["Importance"] * 100

    st.write(
        f"**{row['Feature']}** — "
        f"{percentage:.2f}%"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictive Maintenance System • "
    "Random Forest Machine Learning Model"
)