import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import requests
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# Register premium font (Helvetica is available, or use a custom one if available)
try:
    pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'Helvetica-Bold.ttf'))
except:
    pass  # Fallback to default

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

# Ensure data types are correct
data['square_feet'] = pd.to_numeric(data['square_feet'], errors='coerce')
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['num_rooms'] = pd.to_numeric(data['num_rooms'], errors='coerce')
data['distance_to_city(km)'] = pd.to_numeric(data['distance_to_city(km)'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')

# Drop rows with NaN values in critical columns
data = data.dropna(subset=['square_feet', 'age', 'num_rooms', 'distance_to_city(km)', 'price'])

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

# Sidebar input features for prediction
st.sidebar.subheader("Prediction Inputs")
square_feet = st.sidebar.number_input("Square Feet", 
                                     min_value=500.0, 
                                     max_value=5000.0, 
                                     value=2000.0, 
                                     step=100.0)
num_rooms = st.sidebar.slider("Number of Rooms", 
                              min_value=1, 
                              max_value=10, 
                              value=3)
age = st.sidebar.slider("Age of House (years)", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=20.0)
distance_to_city = st.sidebar.number_input("Distance to City Center (km)", 
                                          min_value=0.0, 
                                          max_value=50.0, 
                                          value=10.0, 
                                          step=1.0)

# Sidebar visualization options
st.sidebar.subheader("Visualizations")
plot_type = st.sidebar.selectbox("Select Plot Type", ["None", "Price Distribution", "Feature Correlation", "Scatter Plot"])
feature_for_scatter = st.sidebar.selectbox("Select Feature for Scatter Plot", 
                                         ["square_feet", "num_rooms", "age", "distance_to_city(km)"], 
                                         index=0)

# Use original data (no filters applied)
filtered_data = data

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

# Create tabs
tab1, tab2 = st.tabs(["Prediction", "Visualizations"])

# Tab 1: Prediction
with tab1:
    # Main content with interactive effects
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Model R² Score", f"{score:.4f}", delta=None, delta_color="normal")
        st.progress(score)

    with col2:
        st.write("""
        This app predicts house prices based on square footage, number of rooms, age of the house, and distance to the city center.
        Adjust the inputs in the sidebar to get a predicted price.
        """)

    # Live prediction
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

    # Download PDF button
    @st.cache_data
    def generate_pdf_report(model_type, score, square_feet, num_rooms, age, distance_to_city, prediction):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom style for premium look
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.black,
            spaceAfter=30,
            alignment=1  # Center
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.black,
            spaceAfter=12
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            textColor=colors.black,
            spaceAfter=6
        )
        
        story = []
        
        # Title
        story.append(Paragraph("House Price Prediction Report", title_style))
        story.append(Spacer(1, 12))
        
        # Model Details
        story.append(Paragraph("Model Details", heading_style))
        model_data = [
            ['Model Type', model_type],
            ['R² Score', f"{score:.4f}"]
        ]
        model_table = Table(model_data)
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(model_table)
        story.append(Spacer(1, 12))
        
        # Input Features
        story.append(Paragraph("Input Features", heading_style))
        input_data = [
            ['Feature', 'Value'],
            ['Square Feet', f"{square_feet:,.0f}"],
            ['Number of Rooms', str(num_rooms)],
            ['Age (years)', f"{age:.0f}"],
            ['Distance to City (km)', f"{distance_to_city:.1f}"]
        ]
        input_table = Table(input_data)
        input_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(input_table)
        story.append(Spacer(1, 12))
        
        # Prediction
        story.append(Paragraph("Prediction Result", heading_style))
        pred_data = [
            ['Predicted Price', f"${prediction:,.2f}"]
        ]
        pred_table = Table(pred_data)
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(pred_table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    pdf_buffer = generate_pdf_report(model_type, score, square_feet, num_rooms, age, distance_to_city, prediction)
    st.download_button(
        label="Download Final Report as PDF",
        data=pdf_buffer,
        file_name="house_price_prediction_report.pdf",
        mime="application/pdf"
    )

# Tab 2: Visualizations
with tab2:
    # Interactive Visualizations with Plotly
    if plot_type != "None":
        st.subheader("Data Visualization")
        if plot_type == "Price Distribution":
            fig = px.histogram(filtered_data, x='price', nbins=30, marginal="rug",
                            title="Price Distribution",
                            labels={'price': 'Price ($)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Feature Correlation":
            corr = filtered_data[['square_feet', 'num_rooms', 'age', 'distance_to_city(km)', 'price']].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                            title="Feature Correlation Matrix",
                            color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Scatter Plot":
            fig = px.scatter(filtered_data, x=feature_for_scatter, y='price',
                            title=f"Price vs {feature_for_scatter.replace('_', ' ').title()}",
                            labels={feature_for_scatter: feature_for_scatter.replace('_', ' ').title(),
                                    'price': 'Price ($)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
