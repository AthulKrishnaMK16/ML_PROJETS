import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import requests
from io import StringIO
import os

# Streamlit app
st.title("House Price Prediction")

# Sidebar
st.sidebar.header("Data and Model Options")

# GitHub URL input
st.sidebar.subheader("Data Source")
github_url = st.sidebar.text_input("GitHub Raw CSV URL (leave blank to try local file)", 
                                  placeholder="https://raw.githubusercontent.com/username/repository/main/house_prices_dataset.csv")

# Load CSV file
data = None
if github_url:
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Check for HTTP errors
        data = pd.read_csv(StringIO(response.text))
        st.session_state['data'] = data
        st.sidebar.success("CSV loaded from GitHub successfully!")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error loading from GitHub: {e}. Trying local file...")
    except pd.errors.ParserError:
        st.sidebar.error("Error parsing GitHub CSV. Please ensure the URL points to a valid CSV. Trying local file...")

# Fallback to local file if GitHub fails or no URL provided
if data is None:
    try:
        data = pd.read_csv('house_prices_dataset.csv')
        st.session_state['data'] = data
        st.sidebar.success("Local CSV loaded successfully!")
    except FileNotFoundError:
        st.error("Error: 'house_prices_dataset.csv' not found locally, and no valid GitHub URL provided. Please provide a valid GitHub URL or place the CSV file in the directory.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error parsing local CSV file. Please ensure 'house_prices_dataset.csv' is a valid CSV.")
        st.stop()

# Model selection
model_type = st.sidebar.selectbox("Select Model", ["Random Forest", "Linear Regression", "Gradient Boosting"])

# Number of estimators for tree-based models
if model_type in ["Random Forest", "Gradient Boosting"]:
    n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
else:
    n_estimators = None  # Not used for Linear Regression

# Scaling option
use_scaling = st.sidebar.checkbox("Use Feature Scaling", value=True)

# Test size for train-test split
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

# Sidebar filters
st.sidebar.subheader("Filter Data")
min_square_feet = st.sidebar.slider("Minimum Square Feet", 
                                   min_value=float(data['square_feet'].min()), 
                                   max_value=float(data['square_feet'].max()), 
                                   value=float(data['square_feet'].min()))
max_age = st.sidebar.slider("Maximum House Age (years)", 
                           min_value=0, 
                           max_value=int(data['age'].max()), 
                           value=int(data['age'].max()))
min_rooms = st.sidebar.slider("Minimum Number of Rooms", 
                             min_value=1, 
                             max_value=int(data['num_rooms'].max()), 
                             value=1)

# Filter data based on sidebar inputs
filtered_data = data[
    (data['square_feet'] >= min_square_feet) &
    (data['age'] <= max_age) &
    (data['num_rooms'] >= min_rooms)
]

# Sidebar visualization options
st.sidebar.subheader("Visualizations")
plot_type = st.sidebar.selectbox("Select Plot Type", ["None", "Price Distribution", "Feature Correlation", "Scatter Plot"])
feature_for_scatter = st.sidebar.selectbox("Select Feature for Scatter Plot", 
                                         ["square_feet", "num_rooms", "age", "distance_to_city(km)"], 
                                         index=0)

# Data preprocessing
data = data[data['price'] > 0]  # Remove negative prices
X = data[['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Scale features if enabled
if use_scaling:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    scaler = None
    X_train_scaled = X_train
    X_test_scaled = X_test

# Train model
if model_type == "Random Forest":
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
elif model_type == "Linear Regression":
    model = LinearRegression()
else:  # Gradient Boosting
    model = XGBRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
score = model.score(X_test_scaled, y_test)
st.write(f"Model R² Score ({model_type}): {score:.4f}")

# Main content
st.write("""
This app predicts house prices based on square footage, number of rooms, age of the house, and distance to the city center.
Enter the details below to get a predicted price.
""")

# Input form
with st.form("prediction_form"):
    square_feet = st.number_input("Square Feet", min_value=500.0, max_value=5000.0, value=2000.0, step=100.0)
    num_rooms = st.slider("Number of Rooms", min_value=1, max_value=10, value=3)
    age = st.slider("Age of House (years)", min_value=0, max_value=100, value=20)
    distance_to_city = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    submitted = st.form_submit_button("Predict Price")

# Prediction
if submitted:
    input_data = pd.DataFrame({
        'square_feet': [square_feet],
        'num_rooms': [num_rooms],
        'age': [age],
        'distance_to_city(km)': [distance_to_city]
    })
    if use_scaling and scaler is not None:
        input_data_scaled = scaler.transform(input_data)
    else:
        input_data_scaled = input_data
    prediction = model.predict(input_data_scaled)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")

# Display filtered dataset
st.subheader("Filtered Data Sample")
st.dataframe(filtered_data.head())

# Display dataset statistics
st.subheader("Filtered Dataset Statistics")
st.write(filtered_data.describe())

# Visualizations
if plot_type != "None":
    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    if plot_type == "Price Distribution":
        sns.histplot(filtered_data['price'], bins=30, kde=True, ax=ax)
        ax.set_title("Price Distribution")
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Count")
    elif plot_type == "Feature Correlation":
        corr = filtered_data[['square_feet', 'num_rooms', 'age', 'distance_to_city(km)', 'price']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Feature Correlation Matrix")
    elif plot_type == "Scatter Plot":
        sns.scatterplot(x=filtered_data[feature_for_scatter], y=filtered_data['price'], ax=ax)
        ax.set_title(f"Price vs {feature_for_scatter.replace('_', ' ').title()}")
        ax.set_xlabel(feature_for_scatter.replace('_', ' ').title())
        ax.set_ylabel("Price ($)")
    st.pyplot(fig)