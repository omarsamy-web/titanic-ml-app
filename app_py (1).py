```python
import streamlit as st
import pandas as pd
import joblib


# Load the trained model
try:
    model = joblib.load("titanicmodel.pkl")
except FileNotFoundError:
    st.error("titanicmodel.pkl not found in the repository.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)


# Title
st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter passenger information to predict survival."
)


# Inputs
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
    0,
    100,
    30
)

sibsp = st.slider(
    "Siblings / Spouses Aboard",
    0,
    8,
    0
)

parch = st.slider(
    "Parents / Children Aboard",
    0,
    6,
    0
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


# Prediction
if st.button("Predict Survival"):

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

        # Make prediction
        prediction = model.predict(input_data)

        # Get probability
        probability = model.predict_proba(input_data)

        if prediction[0] == 1:

            st.success("🎉 Passenger Survived!")

            st.write(
                f"Survival Probability: "
                f"{probability[0][1]:.2%}"
            )

        else:

            st.error("❌ Passenger Did Not Survive.")

            st.write(
                f"Survival Probability: "
                f"{probability[0][0]:.2%}"
            )

    except Exception as e:

        st.error("Prediction Error:")
        st.code(str(e))


# Model information
st.write("---")

st.subheader("Model Details")

st.write("Model: KNN")
st.write("Dataset: Titanic")
st.write("Best Model: Before PCA")
st.write("Accuracy: 0.8212")
```
