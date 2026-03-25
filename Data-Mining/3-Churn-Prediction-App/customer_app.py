import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")


st.title("📊 Customer Churn Prediction App")
st.write(
    "This app uses a Naive Bayes model to predict the probability of a customer leaving the company."
)

try:
    df =pd.read_excel('Data-Mining/3-Churn-Prediction-App/churn_dataset.xlsx')
    model = joblib.load('Data-Mining/3-Churn-Prediction-App/my_model.pkl')
except Exception as e:
    st.error(f"Error: Make sure 'churn_dataset.xlsx' and 'my_model.pkl' exist. {e}")


st.subheader(" Dataset Preview")
st.write(df.head())

st.sidebar.header("Input Customer Features")

age = st.sidebar.slider(
    "Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].mean())
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    int(df["Tenure"].min()),
    int(df["Tenure"].max()),
    int(df["Tenure"].mean()),
)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

gender_n = 1 if gender == "Male" else 0


input_data = np.array([[age, tenure, gender_n]])

if st.sidebar.button("Predict Status"):
    prediction = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Prediction Result")
        if prediction == 1:
            st.error("**Result:** Customer is likely to CHURN")
        else:
            st.success("**Result:** Customer is likely to STAY")

    with col2:
        st.subheader("📈 Prediction Probabilities")
        st.write(f"**Stay Probability:** {pred_proba[0]:.2%}")
        st.write(f"**Churn Probability:** {pred_proba[1]:.2%}")

        st.metric(
            label="Churn Risk",
            value=f"{pred_proba[1]:.2%}",
            delta="High" if prediction == 1 else "Low",
        )
else:
    st.info("Click 'Predict Status' in the sidebar to see the results.")
