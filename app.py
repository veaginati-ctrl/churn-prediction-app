import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 🔥 MUST BE FIRST
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# ---------------- USER DB ----------------
USER_DB = {
    "admin": {"password": "1234", "role": "admin"}
}

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

# ---------------- LOGIN ----------------
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_DB and USER_DB[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = USER_DB[username]["role"]
            st.session_state.username = username
            st.success("Login successful ✅")
            st.rerun()   # 🔥 auto switch
        else:
            st.error("Invalid credentials ❌")

# ---------------- SIGNUP ----------------
def signup():
    st.title("📝 Signup")

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if new_user in USER_DB:
            st.warning("User already exists ⚠️")
        else:
            USER_DB[new_user] = {"password": new_pass, "role": "user"}
            st.success("User created successfully ✅")

# ---------------- MENU ----------------
menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])

# 🔥 BLOCK APP UNTIL LOGIN
if not st.session_state.logged_in:
    if menu == "Login":
        login()
    else:
        signup()
    st.stop()   # 🔥 VERY IMPORTANT

# ---------------- LOAD DATA ----------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("Customer-Churn-Records.csv")

# ---------------- HEADER ----------------
st.title("📊 Customer Churn Prediction Dashboard")

st.sidebar.write(f"👤 {st.session_state.username}")
st.sidebar.write(f"Role: {st.session_state.role}")

# ---------------- LOGOUT ----------------
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.rerun()

# ---------------- NAVIGATION ----------------
if st.session_state.role == "admin":
    option = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])
else:
    option = "Prediction"
    st.sidebar.info("User Mode: Prediction only")

# ---------------- DASHBOARD ----------------
if option == "Dashboard":

    st.subheader("📌 Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['Exited'].mean()*100:.2f}%")
    col3.metric("Avg Age", f"{df['Age'].mean():.1f}")

    st.markdown("---")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Churn Distribution")
        st.bar_chart(df['Exited'].value_counts())

    with col2:
        st.subheader("📈 Age Distribution")
        st.line_chart(df['Age'])

    st.markdown("---")

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)

    st.pyplot(fig)

# ---------------- PREDICTION ----------------
elif option == "Prediction":

    st.subheader("🤖 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 600)

        geo_map = {"France":0, "Germany":1, "Spain":2}
        geo = geo_map[st.selectbox("Geography", geo_map.keys())]

        gender_map = {"Male":1, "Female":0}
        gen = gender_map[st.selectbox("Gender", gender_map.keys())]

        age = st.slider("Age", 18, 100, 30)
        tenure = st.slider("Tenure", 0, 10, 3)

    with col2:
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
        proba = model.predict_proba(data)[0][1]

        st.write(f"Churn Probability: {proba:.2f}")

        if prediction[0] == 1:
            st.error("⚠️ High Risk Customer")
        else:
            st.success("✅ Customer Likely to Stay")