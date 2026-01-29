import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Page Config

st.set_page_config(page_title="Traffic Accident Prediction", layout="wide")

# Load Data & Models
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_traffic_accident_data_3.csv")
    df.columns = df.columns.str.strip()  # remove spaces
    return df

@st.cache_resource
def load_models():
    severity_model = pickle.load(open("accident_severity_model.pkl", "rb"))
    high_risk_model = pickle.load(open("high_risk_model.pkl", "rb"))
    return severity_model, high_risk_model

df = load_data()
severity_model, high_risk_model = load_models()

# Prepare Label Encoders

categorical_cols = ["region_name", "region_type"]
encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

# Helper Function to Prepare Inputs

def prepare_input(input_data, model):
    X = pd.DataFrame([input_data])
    # Ensure all model columns are present
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]
    return X

# Sidebar Navigation

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Project Overview", "Data Analysis", "Accident Severity Prediction", "High Risk Prediction"]
)

# PAGE 1: PROJECT OVERVIEW

if page == "Project Overview":
    st.title("🚦 Traffic Accident Data Science Project")
    st.markdown("""
    ### Objective
    Predict **Accident Severity** and identify **High-Risk Regions** using Machine Learning.
    
    ### Tech Stack
    - Python
    - Pandas & NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib / Seaborn

    ### Models
    - Random Forest Classifier for Accident Severity
    - Random Forest Classifier for High-Risk Prediction

    **How to Use:**  
    1. Navigate to "Accident Severity Prediction" to predict severity.  
    2. Navigate to "High Risk Prediction" to check if a region is high risk.  
    3. Use the "Data Analysis" page to explore trends.
    """)

# PAGE 2: DATA ANALYSIS / GRAPHS

if page == "Data Analysis":
    st.title("📊 Traffic Accident Data Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Traffic Accidents vs Deaths")
        fig, ax = plt.subplots()
        if "total_traffic_accidents_cases" in df.columns and "total_traffic_accidents_died" in df.columns:
            ax.plot(df["total_traffic_accidents_cases"], label="Cases", color="blue")
            ax.plot(df["total_traffic_accidents_died"], label="Deaths", color="red")
        ax.set_xlabel("Index")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Road Accidents Injured Distribution")
        if "road_accidents_injured" in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df["road_accidents_injured"].dropna(), bins=20, color="green", alpha=0.7)
            ax.set_xlabel("Injured Count")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

# PAGE 3: ACCIDENT SEVERITY PREDICTION

if page == "Accident Severity Prediction":
    st.title("⚠️ Accident Severity Prediction")

    input_data = {}
    model_features = severity_model.feature_names_in_

    st.markdown("### Enter the Accident Data:")

    for col in model_features:
        if col in df.select_dtypes(include=[np.number]).columns:
            default_val = int(df[col].mean())
            step_val = 1
            input_data[col] = st.number_input(
                col.replace("_", " ").title(),
                value=default_val,
                step=step_val
            )
        elif col in categorical_cols:
            options = df[col].unique().tolist()
            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                options
            )
        else:
            input_data[col] = 0

    if st.button("Predict Severity"):
        # Encode categorical inputs
        for col in categorical_cols:
            if col in input_data:
                input_data[col] = encoders[col].transform([input_data[col]])[0]

        # Prepare input and predict
        X = prepare_input(input_data, severity_model)
        prediction = severity_model.predict(X)[0]
        st.success(f"🚨 Predicted Accident Severity: **{prediction}**")

# PAGE 4: HIGH RISK REGION PREDICTION

if page == "High Risk Prediction":
    st.title("🔥 High Risk Accident Prediction")

    input_data = {}
    model_features = high_risk_model.feature_names_in_

    st.markdown("### Enter Data to Check High Risk:")

    for col in model_features:
        if col in df.select_dtypes(include=[np.number]).columns:
            default_val = int(df[col].mean())
            step_val = 1
            input_data[col] = st.number_input(
                col.replace("_", " ").title(),
                value=default_val,
                step=step_val
            )
        elif col in categorical_cols:
            options = df[col].unique().tolist()
            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                options
            )
        else:
            input_data[col] = 0

    if st.button("Predict Risk"):
        # Encode categorical inputs
        for col in categorical_cols:
            if col in input_data:
                input_data[col] = encoders[col].transform([input_data[col]])[0]

        X = prepare_input(input_data, high_risk_model)
        result = high_risk_model.predict(X)[0]
        if result == 1:
            st.error("🚨 High Risk Accident Area")
        else:
            st.success("✅ Low Risk Accident Area")
