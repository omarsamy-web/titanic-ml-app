```python
import streamlit as st
import pandas as pd
import joblib


# =========================
# Load trained model
# =========================

try:
    model = joblib.load("titanicmodel.pkl")
except FileNotFoundError:
    st.error(
        "Error: titanicmodel.pkl not found. "
        "Make sure the file is uploaded to the GitHub repository."
    )
    st.stop()


# =========================
# Streamlit App
# =========================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter the passenger information below "
    "to predict whether the passenger survived."
)


# =========================
# User Inputs
# =========================

pclass = st.selectbox(
    "Passenger Class (Pclass)",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.slider(
    "Age",
    min_value=0,
    max_value=100,
    value=30
)

sibsp = st.slider(
    "Siblings / Spouses Aboard (SibSp)",
    min_value=0,
    max_value=8,
    value=0
)

parch = st.slider(
    "Parents / Children Aboard (Parch)",
    min_value=0,
    max_value=6,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0,
    step=0.1
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["C", "Q", "S"]
)


# =========================
# Prediction
# =========================

if st.button("🔮 Predict Survival"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Pclass": [pclass],
        "Sex": [sex],
        "Embarked": [embarked]
    })

    try:

        # Prediction
        prediction = model.predict(input_data)

        # Probability
        prediction_proba = model.predict_proba(input_data)

        # =========================
        # Display Result
        # =========================

        if prediction[0] == 1:

            probability = prediction_proba[0][1]

            st.success(
                f"🎉 Prediction: Survived!\n\n"
                f"Survival Probability: {probability:.2%}"
            )

        else:

            probability = prediction_proba[0][0]

            st.error(
                f"❌ Prediction: Not Survived.\n\n"
                f"Survival Probability: {probability:.2%}"
            )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.code(str(e))


# =========================
# Model Information
# =========================

st.write("---")

st.subheader("📊 Model Details")

st.write("Model: K-Nearest Neighbors (KNN)")
st.write("Dataset: Titanic")
st.write("Best Model: Before PCA")
st.write("Accuracy: 0.8212")
```
