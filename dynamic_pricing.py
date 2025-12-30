import streamlit as st
from model import predict_price, accuracy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Dynamic Pricing Predictor")
st.write("Predict ride prices using machine learning")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Price Prediction", "Data Analysis"])

if page == "Price Prediction":
    st.header("Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        riders = st.slider("Number of Riders", 20, 100, 50)
        drivers = st.slider("Number of Drivers", 5, 100, 30)
        past_rides = st.slider("Past Rides", 0, 100, 20)

    with col2:
        rating = st.slider("Average Rating", 3.0, 5.0, 4.0, 0.1)
        duration = st.slider("Ride Duration (minutes)", 10, 180, 60)

    if st.button("Predict Price"):
        prediction = predict_price(riders, drivers, past_rides, rating, duration)
        st.success(f"Predicted Price: ₹{prediction:.2f}")

elif page == "Data Analysis":
    st.header("Data Analysis")

    df = pd.read_csv('dynamic_pricing (2).csv')

    # Show model accuracy in data analysis section
    st.metric("Model Accuracy", f"{accuracy:.1%}")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    # Key metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rides", len(df))
    with col2:
        st.metric("Average Price", f"₹{df['Historical_Cost_of_Ride'].mean():.0f}")

    # Visualizations using matplotlib
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['Historical_Cost_of_Ride'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Price (₹)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Ride Prices')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Scatter plot for riders vs price
    st.subheader("Riders vs Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Number_of_Riders'], df['Historical_Cost_of_Ride'], alpha=0.6, color='blue')
    ax.set_xlabel('Number of Riders')
    ax.set_ylabel('Price (₹)')
    ax.set_title('Relationship between Number of Riders and Price')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Duration vs price
    st.subheader("Duration vs Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Expected_Ride_Duration'], df['Historical_Cost_of_Ride'], alpha=0.6, color='green')
    ax.set_xlabel('Ride Duration (minutes)')
    ax.set_ylabel('Price (₹)')
    ax.set_title('Relationship between Ride Duration and Price')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
