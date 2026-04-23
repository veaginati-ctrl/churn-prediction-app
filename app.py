import streamlit as st
import numpy as np
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", 300, 900, 600)

geo_map = {"France":0, "Germany":1, "Spain":2}
gender_map = {"Male":1, "Female":0}

geo = geo_map[st.selectbox("Geography", geo_map.keys())]
gen = gender_map[st.selectbox("Gender", gender_map.keys())]

age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 200000.0, 50000.0)

products = st.slider("Number of Products", 1, 4, 2)
card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])

salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict"):
    data = np.array([[credit_score, geo, gen, age, tenure,
                      balance, products, card, active, salary,
                      0, 0, 0, 0]])

    data = scaler.transform(data)
    prediction = model.predict(data)

    # ✅ Probability
    proba = model.predict_proba(data)[0][1]
    st.write(f"Churn Probability: {proba:.2f}")

    # ✅ Final Output
    if prediction[0] == 1:
        st.error("High Risk Customer ⚠️")
    else:
        st.success("Customer likely to stay 👍")