# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv('tripadvisor_review.csv')

# Sidebar Navigation
st.sidebar.title("Travel Reviews App")
page = st.sidebar.radio("Go to", ["Home", "Data Preview", "Predict", "Report"])

# Home Page
if page == "Home":
    st.title("Travel Reviews Predictor App")
    st.write("""
    Welcome! This app uses a trained Random Forest model to predict an average travel review score 
    based on user inputs. Explore data, make predictions, and view insights!
    """)

# Data Preview Page
elif page == "Data Preview":
    st.title("Data Preview")
    st.write("Here's a sample of the dataset:")
    st.dataframe(df.head(20))
    
    st.write("Category-wise statistics:")
    st.dataframe(df.describe())

    st.write("Category Distributions:")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.drop(['User ID'], axis=1).hist(ax=ax)
    st.pyplot(fig)

# Prediction Page
elif page == "Predict":
    st.title("Predict Average Review Score")
    st.write("Enter the ratings for each category:")

    input_data = []
    for i in range(1, 11):
        value = st.number_input(f"Category {i} rating", min_value=0.0, max_value=5.0, step=0.1, value=2.5)
        input_data.append(value)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        st.success(f"Predicted Average Review Score: {prediction[0]:.2f}")

# Report Page
elif page == "Report":
    st.title("Model Performance Report")
    st.write("The model was evaluated with the following metrics:")

    st.markdown("""
    - **Algorithm:** Random Forest Regressor
    - **RMSE:** 0.051 (Low error, indicating high accuracy)
    - **R² Score:** 0.903 (Model explains 90.3% of the variance)
    """)

    st.write("Feature Importance:")
    importances = model.feature_importances_
    categories = [f"Category {i}" for i in range(1, 11)]
    importance_df = pd.DataFrame({'Feature': categories, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.dataframe(importance_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.invert_yaxis()
    st.pyplot(fig)
