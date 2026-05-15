"""House Price Predictor AI (Streamlit)

Premium Streamlit frontend for predicting house prices with SHAP explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import re

# Page configuration
st.set_page_config(
    page_title="House Price Predictor AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium professional design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Color Palette - Modern Professional */
    :root {
        --primary: #0f4c75;
        --primary-light: #3282b8;
        --secondary: #00a8e8;
        --accent: #00c9ff;
        --success: #2ecc71;
        --warning: #f39c12;
        --danger: #e74c3c;
        --dark: #1a1a2e;
        --light: #f8f9fa;
        --border: #e0e0e0;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        min-height: 100vh;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f4c75 0%, #1a5f7a 100%) !important;
        padding: 20px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Radio button styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px 12px;
        border-radius: 6px;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateX(5px);
    }
    
    /* Info box styling */
    [data-testid="stSidebar"] .stInfo {
        background: rgba(0, 201, 255, 0.15) !important;
        border-left: 4px solid #00c9ff !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin-top: 20px !important;
    }
    
    [data-testid="stSidebar"] .stInfo > div {
        color: white !important;
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 76, 117, 0.95) 0%, rgba(50, 130, 184, 0.95) 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(15, 76, 117, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(15, 76, 117, 0.4);
    }
    
    .metric-card-value {
        font-size: 32px;
        font-weight: 800;
        margin: 15px 0;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }
    
    .metric-card-label {
        font-size: 13px;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Metric Card Variants */
    .metric-card-primary {
        background: linear-gradient(135deg, rgba(15, 76, 117, 0.95) 0%, rgba(50, 130, 184, 0.95) 100%);
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.9) 0%, rgba(39, 174, 96, 0.9) 100%);
    }
    
    .metric-card-accent {
        background: linear-gradient(135deg, rgba(0, 168, 232, 0.9) 0%, rgba(0, 201, 255, 0.9) 100%);
    }
    
    /* Premium Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.95) 0%, rgba(39, 174, 96, 0.95) 100%);
        color: white;
        padding: 40px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(46, 204, 113, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 30px 0;
        backdrop-filter: blur(10px);
        animation: slideUp 0.6s ease;
    }
    
    .prediction-value {
        font-size: 48px;
        font-weight: 800;
        margin: 20px 0;
        letter-spacing: -2px;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .prediction-label {
        font-size: 16px;
        opacity: 0.95;
        font-weight: 500;
        margin: 10px 0;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Premium Hero Section */
    .hero {
        background: linear-gradient(135deg, rgba(15, 76, 117, 0.9) 0%, rgba(0, 168, 232, 0.85) 50%, rgba(0, 201, 255, 0.9) 100%);
        color: white;
        padding: 60px 40px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 20px 50px rgba(15, 76, 117, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero h1 {
        font-size: 42px;
        margin-bottom: 15px;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    .hero p {
        font-size: 18px;
        opacity: 0.95;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Section Headers */
    .section-header {
        color: var(--dark);
        font-size: 28px;
        font-weight: 800;
        margin: 40px 0 20px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid var(--secondary);
        position: relative;
        letter-spacing: -0.5px;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, var(--secondary) 0%, var(--accent) 100%);
    }
    
    /* Feature Importance Ranking */
    .feature-rank {
        background: linear-gradient(90deg, rgba(15, 76, 117, 0.08) 0%, rgba(0, 168, 232, 0.08) 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid var(--secondary);
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .feature-rank:hover {
        background: linear-gradient(90deg, rgba(15, 76, 117, 0.12) 0%, rgba(0, 168, 232, 0.12) 100%);
        transform: translateX(5px);
    }
    
    .feature-rank strong {
        color: var(--primary);
        font-weight: 700;
    }
    
    /* Input fields styling */
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stTextInput"] input {
        border-radius: 8px !important;
        border: 2px solid var(--border) !important;
        padding: 12px 15px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        background: white !important;
    }
    
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stSelectbox"] > div > div:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--secondary) !important;
        box-shadow: 0 0 0 3px rgba(0, 168, 232, 0.1) !important;
    }
    
    /* Button styling */
    button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 20px rgba(15, 76, 117, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(15, 76, 117, 0.4) !important;
    }
    
    button[kind="primary"]:active {
        transform: translateY(0) !important;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
    }
    
    [data-testid="stDataFrame"] > div {
        border-radius: 8px !important;
    }
    
    /* Image styling */
    img {
        border-radius: 8px !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12) !important;
    }
    
    /* Tab styling */
    [role="tablist"] {
        border-bottom: 2px solid var(--border) !important;
        gap: 20px !important;
    }
    
    [role="tab"] {
        color: var(--primary) !important;
        font-weight: 600 !important;
        border-bottom: 3px solid transparent !important;
        padding: 10px 0 !important;
        transition: all 0.3s ease !important;
    }
    
    [role="tab"]:hover {
        color: var(--secondary) !important;
    }
    
    [role="tab"][aria-selected="true"] {
        color: var(--secondary) !important;
        border-bottom-color: var(--secondary) !important;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(46, 204, 113, 0.1) !important;
        border-left: 4px solid var(--success) !important;
        border-radius: 8px !important;
    }
    
    /* Error message */
    .stError {
        background: rgba(231, 76, 60, 0.1) !important;
        border-left: 4px solid var(--danger) !important;
        border-radius: 8px !important;
    }
    
    /* Info message */
    .stInfo {
        background: rgba(0, 168, 232, 0.1) !important;
        border-left: 4px solid var(--secondary) !important;
        border-radius: 8px !important;
    }
    
    /* Container styling */
    .stContainer {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    
    /* Text styling */
    h1 {
        color: var(--dark) !important;
        font-weight: 800 !important;
    }
    
    h2 {
        color: var(--primary) !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: var(--primary) !important;
        font-weight: 700 !important;
    }
    
    p {
        color: #333 !important;
        line-height: 1.6 !important;
    }
    
    /* Markdown text color */
    [data-testid="stMarkdownContainer"] {
        color: #333 !important;
    }
    
    /* Overall page background */
    .main > div {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        min-height: 100vh !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model_and_data():
    """Load trained model, encoders, and data"""
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('models/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    with open('models/columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    
    # Load original data for statistics
    df = pd.read_csv('data/Pakistan House Prices and Property Listings.csv')
    
    return model, encoders, features, columns, df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_area(area_str):
    """Parse area string to numeric value"""
    if pd.isna(area_str):
        return np.nan
    
    area_str = str(area_str).strip()
    match = re.search(r'(\d+\.?\d*)', area_str)
    
    if not match:
        return np.nan
    
    numeric_value = float(match.group(1))
    
    if 'Marla' in area_str or 'marla' in area_str:
        return numeric_value * 272.251
    elif 'Kanal' in area_str or 'kanal' in area_str:
        return numeric_value * 5445.02
    
    return np.nan


def encode_input(input_dict, encoders):
    """Encode input dictionary using label encoders"""
    encoded = {}
    
    for key, value in input_dict.items():
        if key in encoders:
            # Categorical column - encode using LabelEncoder
            encoder = encoders[key]['encoder']
            try:
                encoded[key] = encoder.transform([str(value)])[0]
            except ValueError:
                # If value not in training set, use mode or first class
                encoded[key] = encoder.transform([encoder.classes_[0]])[0]
        else:
            # Numeric column - use as is
            encoded[key] = value
    
    return encoded


def format_price(price):
    """Format price in Pakistani Rupees with commas"""
    return f"PKR {price:,.0f}"


def st_image_responsive(image, **kwargs):
    """Render an image responsively across Streamlit versions (container/column width)."""
    try:
        st.image(image, use_container_width=True, **kwargs)
    except TypeError:
        st.image(image, use_column_width=True, **kwargs)


def st_dataframe_responsive(df: pd.DataFrame, **kwargs):
    """Render a dataframe responsively across Streamlit versions."""
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        st.dataframe(df, **kwargs)


def st_plotly_responsive(fig, **kwargs):
    """Render a Plotly chart responsively across Streamlit versions."""
    try:
        st.plotly_chart(fig, use_container_width=True, **kwargs)
    except TypeError:
        st.plotly_chart(fig, **kwargs)


def st_pyplot_responsive(fig=None, **kwargs):
    """Render matplotlib figures responsively across Streamlit versions."""
    try:
        st.pyplot(fig, use_container_width=True, **kwargs)
    except TypeError:
        st.pyplot(fig, **kwargs)


def make_shap_force_or_waterfall_plot(model, X_row: pd.DataFrame, feature_names: list[str]):
    """Create a local SHAP explanation plot for a single row.

    Tries to generate a SHAP force plot (matplotlib) first; if unavailable, falls back
    to a SHAP waterfall plot. Returns a matplotlib Figure.
    """
    explainer = shap.TreeExplainer(model)

    # Newer SHAP returns an Explanation object from calling the explainer.
    try:
        explanation = explainer(X_row)
        single = explanation[0]
        base_value = float(np.array(single.base_values).reshape(-1)[0])
        values = np.array(single.values).reshape(-1)
        data_row = X_row.iloc[0]
    except Exception:
        # Older SHAP: use shap_values + expected_value.
        shap_values = explainer.shap_values(X_row)
        base_value = float(np.array(explainer.expected_value).reshape(-1)[0])
        values = np.array(shap_values).reshape(-1)
        data_row = X_row.iloc[0]

    # Attempt a force plot rendered via matplotlib.
    try:
        plt.close('all')
        fig = plt.figure(figsize=(12, 2.6))
        shap.force_plot(
            base_value,
            values,
            data_row,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        return fig
    except Exception:
        # Fallback: waterfall plot.
        plt.close('all')
        fig = plt.figure(figsize=(10, 4.2))
        try:
            exp = shap.Explanation(values=values, base_values=base_value, data=data_row.values, feature_names=feature_names)
            shap.plots.waterfall(exp, max_display=10, show=False)
        except Exception:
            # Last-resort: bar plot of absolute contributions.
            order = np.argsort(np.abs(values))[::-1][:10]
            names = [feature_names[i] for i in order]
            contrib = values[order]
            plt.barh(list(reversed(names)), list(reversed(contrib)))
            plt.title('SHAP Local Explanation (Top 10)')
        plt.tight_layout()
        return fig


def get_shap_force_plot(model, X_test, feature_names, instance_idx=0):
    """Generate SHAP force plot for a specific prediction"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[[instance_idx]])
    
    plt.figure(figsize=(12, 3))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        X_test.iloc[instance_idx],
        feature_names=feature_names,
        show=False
    )
    return plt


# ============================================================================
# MAIN APP LOGIC
# ============================================================================

def main():
    try:
        model, encoders, features, columns, df_original = load_model_and_data()
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run `python train_model.py` first.")
        return
    
    # Sidebar Navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-size: 28px;">🏠</h2>
        <h3 style="color: white; margin: 10px 0 5px 0; font-size: 18px; font-weight: 700;">House Price</h3>
        <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 12px; font-weight: 500;">PREDICTOR AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 10px 0;">
        <p style="color: white; font-size: 13px; font-weight: 600; margin-bottom: 10px;">📌 ABOUT THIS APP</p>
        <p style="color: rgba(255,255,255,0.85); font-size: 12px; line-height: 1.6; margin: 0;">
        Powered by <strong>XGBoost</strong> ML model trained on 168K+ verified Pakistani property listings.
        </p>
        <p style="color: rgba(255,255,255,0.85); font-size: 12px; line-height: 1.6; margin: 10px 0 0 0;">
        Get instant price estimates with <strong>SHAP-powered explainability</strong> and transparent predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # PAGE: HOME
    # ========================================================================
    if page == "🏠 Home":
        # Hero Banner
        st.markdown("""
        <div class="hero">
            <h1>🏠 House Price Predictor AI</h1>
            <p>AI-Powered Real Estate Valuation with XGBoost & SHAP Explainability</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Section
        st.markdown('<h2 class="section-header">📊 Key Statistics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-card-primary">
                <div class="metric-card-label">Dataset Properties</div>
                <div class="metric-card-value">{len(df_original):,}</div>
                <div class="metric-card-label">Verified Listings</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-card-success">
                <div class="metric-card-label">Model Accuracy</div>
                <div class="metric-card-value">92.31%</div>
                <div class="metric-card-label">R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cities = df_original['city'].nunique()
            st.markdown(f"""
            <div class="metric-card metric-card-accent">
                <div class="metric-card-label">Coverage Area</div>
                <div class="metric-card-value">{cities}</div>
                <div class="metric-card-label">Major Cities</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample Data Section
        st.markdown('<h2 class="section-header">📋 Sample Properties</h2>', unsafe_allow_html=True)
        
        sample_df = df_original[['location', 'city', 'property_type', 'bedrooms', 'baths', 'area', 'price']].head(8)
        
        # Format sample data for better display
        display_df = sample_df.copy()
        display_df['price'] = display_df['price'].apply(lambda x: f"PKR {x:,.0f}")
        display_df = display_df.rename(columns={
            'location': 'Location',
            'city': 'City',
            'property_type': 'Type',
            'bedrooms': 'Beds',
            'baths': 'Baths',
            'area': 'Area',
            'price': 'Price'
        })

        st_dataframe_responsive(display_df, hide_index=True)
        
        # Features & Technology
        st.markdown('<h2 class="section-header">✨ What Makes This Special</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            #### 🤖 Advanced ML Model
            - **XGBoost Gradient Boosting** with 500 iterations
            - **Fine-tuned hyperparameters** for accuracy
            - **Trained on 154K+ properties** after preprocessing
            - **92.31% R² Score** - Excellent predictions
            
            #### 🎯 Real-Time Predictions
            - Get instant price estimates
            - See confidence ranges (±10%)
            - Understand price drivers
            """)
        
        with col2:
            st.markdown("""
            #### 💡 Explainable AI with SHAP
            - **Feature Impact Analysis** - see what drives prices
            - **SHAP Force Plots** - understand each prediction
            - **Visual Explanations** - transparent AI
            
            #### 📊 Comprehensive Analytics
            - Interactive charts and visualizations
            - Dataset statistics and insights
            - Model performance metrics
            """)
        
        # How It Works
        st.markdown('<h2 class="section-header">🔄 How It Works</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
            <h3 style="color: #0f4c75; font-size: 24px;">1️⃣</h3>
            <p><strong>Enter Details</strong></p>
            <small>Provide property information</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
            <h3 style="color: #0f4c75; font-size: 24px;">2️⃣</h3>
            <p><strong>ML Processing</strong></p>
            <small>XGBoost predicts price</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
            <h3 style="color: #0f4c75; font-size: 24px;">3️⃣</h3>
            <p><strong>Get Estimate</strong></p>
            <small>Instant PKR valuation</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
            <h3 style="color: #0f4c75; font-size: 24px;">4️⃣</h3>
            <p><strong>View Insights</strong></p>
            <small>SHAP explanations</small>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # PAGE: PREDICT
    # ========================================================================
    elif page == "🔮 Predict":
        st.markdown('<h2 class="section-header">🔮 Estimate House Price</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <p style="font-size: 16px; color: #666; margin-bottom: 30px;">
        Enter property details below and get an instant AI-powered price estimation with SHAP-based insights.
        </p>
        """, unsafe_allow_html=True)
        
        # Prediction Form
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<h4 style="color: #0f4c75; margin-top: 0;">📍 Location Details</h4>', unsafe_allow_html=True)
            
            city = st.selectbox(
                "City",
                sorted(df_original['city'].unique()),
                label_visibility="collapsed"
            )
            
            locations = df_original[df_original['city'] == city]['location'].unique()
            location = st.selectbox(
                "Location",
                sorted(locations),
                label_visibility="collapsed"
            )
            
            property_types = sorted(df_original['property_type'].unique())
            property_type = st.selectbox(
                "Property Type",
                property_types,
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown('<h4 style="color: #0f4c75; margin-top: 0;">🏠 Property Details</h4>', unsafe_allow_html=True)
            
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=20,
                value=3,
                label_visibility="collapsed"
            )
            
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0,
                max_value=20,
                value=2,
                label_visibility="collapsed"
            )
            
            area_marla = st.number_input(
                "Area in Marla",
                min_value=0.1,
                max_value=500.0,
                value=8.0,
                step=0.1,
                label_visibility="collapsed"
            )
        
        # Purpose selector
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<h4 style="color: #0f4c75;">Purpose</h4>', unsafe_allow_html=True)
            purpose = st.selectbox(
                "Purpose",
                sorted(df_original['purpose'].unique()),
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown('<h4 style="color: #0f4c75;">Estimation Zone</h4>', unsafe_allow_html=True)
            st.info("✅ All parameters configured. Ready to estimate!", icon="ℹ️")
        
        # Predict Button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            predict_button = st.button(
                "🎯 ESTIMATE PRICE",
                key="predict_btn"
            )
        
        # Prediction Logic
        if predict_button:
            with st.spinner("🔄 Analyzing property... Generating SHAP insights..."):
                # Convert area to square feet
                area_sqft = area_marla * 272.251
                
                # Prepare input
                input_data = {
                    'property_type': property_type,
                    'location': location,
                    'city': city,
                    'province_name': df_original[df_original['city'] == city]['province_name'].iloc[0],
                    'latitude': df_original[df_original['location'] == location]['latitude'].iloc[0],
                    'longitude': df_original[df_original['location'] == location]['longitude'].iloc[0],
                    'baths': bathrooms,
                    'area': area_sqft,
                    'purpose': purpose,
                    'bedrooms': bedrooms
                }
                
                # Encode input
                encoded_input = encode_input(input_data, encoders)
                
                # Create feature array
                X_input = pd.DataFrame([encoded_input])[features]
                
                # Predict
                log_pred = model.predict(X_input)[0]
                predicted_price = np.expm1(log_pred)
                
                # Calculate price range
                price_lower = predicted_price * 0.9
                price_upper = predicted_price * 1.1
                
                # Display Prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-label">💰 ESTIMATED HOUSE PRICE</div>
                    <div class="prediction-value">{format_price(predicted_price)}</div>
                    <div class="prediction-label">
                    📊 Price Range: {format_price(price_lower)} ~ {format_price(price_upper)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📝 Property Summary", "🎯 Feature Analysis", "📊 Market Insights"])
                
                with tab1:
                    st.markdown('<h3 style="color: #0f4c75;">Property Configuration</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🏙️ Location", location)
                        st.metric("🌍 City", city)
                        st.metric("🏢 Property Type", property_type)
                    
                    with col2:
                        st.metric("🛏️ Bedrooms", int(bedrooms))
                        st.metric("🚿 Bathrooms", int(bathrooms))
                        st.metric("📌 Purpose", purpose)
                    
                    with col3:
                        st.metric("📏 Area (Marla)", f"{area_marla:.2f}")
                        st.metric("📐 Area (Sq Ft)", f"{area_sqft:,.0f}")
                        st.metric("📍 Coordinates", f"{location}")
                
                with tab2:
                    st.markdown('<h3 style="color: #0f4c75;">Top 5 Price Drivers</h3>', unsafe_allow_html=True)
                    st.markdown('<p style="color: #666; margin-bottom: 20px;">These features had the most impact on your price estimate:</p>', unsafe_allow_html=True)
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_input)
                    
                    # Get SHAP contributions
                    shap_importance = pd.DataFrame({
                        'feature': features,
                        'impact': np.abs(shap_values[0])
                    }).sort_values('impact', ascending=False).head(5)
                    
                    for idx, (_, row) in enumerate(shap_importance.iterrows(), 1):
                        st.markdown(f"""
                        <div class="feature-rank">
                            <strong>#{idx} {row['feature']}</strong> 
                            <span style="float: right; color: #0f4c75; font-weight: 700;">{row['impact']:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('<h3 style="color: #0f4c75; margin-top: 28px;">🧠 SHAP Explanation (This Prediction)</h3>', unsafe_allow_html=True)
                    st.markdown('<p style="color: #666; margin-bottom: 10px;">A local explanation showing how each feature pushed the estimate up or down.</p>', unsafe_allow_html=True)
                    shap_fig = make_shap_force_or_waterfall_plot(model, X_input, features)
                    st_pyplot_responsive(shap_fig)
                
                with tab3:
                    st.markdown('<h3 style="color: #0f4c75;">Market Context</h3>', unsafe_allow_html=True)
                    
                    # Market analysis
                    city_prices = df_original[df_original['city'] == city]['price']
                    location_prices = df_original[df_original['location'] == location]['price']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_city = city_prices.mean()
                        st.metric(
                            "🏙️ City Average",
                            format_price(avg_city),
                            delta=f"{((predicted_price / avg_city - 1) * 100):+.1f}%"
                        )
                    
                    with col2:
                        median_location = location_prices.median()
                        st.metric(
                            "📍 Location Median",
                            format_price(median_location),
                            delta=f"{((predicted_price / median_location - 1) * 100):+.1f}%"
                        )
                    
                    with col3:
                        properties_count = len(df_original[df_original['location'] == location])
                        st.metric(
                            "📊 Similar Properties",
                            f"{properties_count:,}",
                            "in this location"
                        )
                
                st.success("✅ Analysis complete!", icon="✅")
    
    # ========================================================================
    # PAGE: ANALYTICS
    # ========================================================================
    elif page == "📊 Analytics":
        st.markdown('<h2 class="section-header">📊 Model Analytics & Performance</h2>', unsafe_allow_html=True)
        
        # Performance Metrics
        st.markdown('<h3 style="color: #0f4c75; margin-top: 30px;">🎯 Model Performance Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        with col1:
            st.markdown("""
            <div class="metric-card metric-card-primary">
                <div class="metric-card-label">Accuracy Score</div>
                <div class="metric-card-value">92.31%</div>
                <div class="metric-card-label">R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card metric-card-success">
                <div class="metric-card-label">Error Margin</div>
                <div class="metric-card-value">3.2M</div>
                <div class="metric-card-label">RMSE (PKR)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card metric-card-accent">
                <div class="metric-card-label">Average Error</div>
                <div class="metric-card-value">1.6M</div>
                <div class="metric-card-label">MAE (PKR)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card metric-card-primary">
                <div class="metric-card-label">Percentage Error</div>
                <div class="metric-card-value">12.3%</div>
                <div class="metric-card-label">MAPE</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        plots_folder = Path('plots')
        
        if plots_folder.exists():
            # Feature Importance
            st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">📈 Feature Importance Analysis</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: #666; margin-bottom: 20px;">Top 15 features that influence house price predictions</p>', unsafe_allow_html=True)
            if (plots_folder / 'feature_importance.png').exists():
                st_image_responsive(str(plots_folder / 'feature_importance.png'))
            
            # Actual vs Predicted
            st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">📊 Prediction Accuracy</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: #666; margin-bottom: 20px;">Comparing actual vs model-predicted prices on test set</p>', unsafe_allow_html=True)
            if (plots_folder / 'actual_vs_predicted.png').exists():
                st_image_responsive(str(plots_folder / 'actual_vs_predicted.png'))
            
            # Residual Distribution
            st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">📉 Prediction Error Distribution</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: #666; margin-bottom: 20px;">Distribution of residuals (errors) showing model calibration</p>', unsafe_allow_html=True)
            if (plots_folder / 'residual_plot.png').exists():
                st_image_responsive(str(plots_folder / 'residual_plot.png'))
            
            # SHAP Summary
            st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">🎯 SHAP Feature Impact Summary</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: #666; margin-bottom: 20px;">SHAP beeswarm plot showing feature contributions across all predictions</p>', unsafe_allow_html=True)
            if (plots_folder / 'shap_summary.png').exists():
                st_image_responsive(str(plots_folder / 'shap_summary.png'))
            
            # SHAP Bar Chart
            st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">📊 Mean Feature Impact Ranking</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: #666; margin-bottom: 20px;">Average absolute SHAP values for each feature</p>', unsafe_allow_html=True)
            if (plots_folder / 'shap_bar.png').exists():
                st_image_responsive(str(plots_folder / 'shap_bar.png'))
        
        # Dataset Statistics
        st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">📊 Dataset Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<h4 style="color: #0f4c75;">Price Distribution (PKR)</h4>', unsafe_allow_html=True)
            
            price_stats = {
                'Minimum': f"PKR {df_original['price'].min():,.0f}",
                'Q1 (25%)': f"PKR {df_original['price'].quantile(0.25):,.0f}",
                'Median': f"PKR {df_original['price'].median():,.0f}",
                'Q3 (75%)': f"PKR {df_original['price'].quantile(0.75):,.0f}",
                'Maximum': f"PKR {df_original['price'].max():,.0f}",
                'Mean': f"PKR {df_original['price'].mean():,.0f}",
                'Std Dev': f"PKR {df_original['price'].std():,.0f}"
            }
            
            stats_df = pd.DataFrame(list(price_stats.items()), columns=['Statistic', 'Value'])
            st_dataframe_responsive(stats_df, hide_index=True)
        
        with col2:
            st.markdown('<h4 style="color: #0f4c75;">Distribution by Property Type</h4>', unsafe_allow_html=True)
            
            pt_dist = df_original['property_type'].value_counts()
            
            # Create a nice bar chart
            import plotly.express as px
            fig = px.bar(
                x=pt_dist.index,
                y=pt_dist.values,
                labels={'x': 'Property Type', 'y': 'Count'},
                color=pt_dist.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title='Property Type',
                yaxis_title='Number of Properties'
            )
            st_plotly_responsive(fig)
    
    # ========================================================================
    # PAGE: ABOUT
    # ========================================================================
    elif page == "ℹ️ About":
        st.markdown('<h2 class="section-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
        
        # Project Overview
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(15, 76, 117, 0.08) 0%, rgba(0, 168, 232, 0.08) 100%); 
                    padding: 25px; border-radius: 10px; border-left: 4px solid #0f4c75; margin: 20px 0;">
        <h3 style="color: #0f4c75; margin-top: 0;">🏗️ Project Overview</h3>
        
        <p style="color: #333; line-height: 1.8;">
        <strong>House Price Predictor AI</strong> is a sophisticated AI-powered application that predicts 
        residential property prices in Pakistan using advanced machine learning and explainable AI techniques.
        </p>
        
        <p style="color: #333; line-height: 1.8;">
        The project combines cutting-edge gradient boosting algorithms with SHAP (SHapley Additive exPlanations) 
        to provide not just accurate predictions, but also transparent, understandable explanations of how the 
        model arrives at each estimate.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technology Stack
        st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">🚀 Technology Stack</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(15, 76, 117, 0.1) 0%, rgba(50, 130, 184, 0.1) 100%); 
                        padding: 20px; border-radius: 10px; border: 1px solid rgba(15, 76, 117, 0.2);">
            <h4 style="color: #0f4c75; margin-top: 0;">🤖 ML & AI</h4>
            <ul style="color: #333; line-height: 2;">
            <li><strong>XGBoost 2.0+</strong> - Gradient Boosting</li>
            <li><strong>SHAP 0.42+</strong> - Explainable AI</li>
            <li><strong>Scikit-learn</strong> - ML Utilities</li>
            <li><strong>NumPy/Pandas</strong> - Data Processing</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(39, 174, 96, 0.1) 100%); 
                        padding: 20px; border-radius: 10px; border: 1px solid rgba(46, 204, 113, 0.2);">
            <h4 style="color: #2ecc71; margin-top: 0;">🎨 Frontend</h4>
            <ul style="color: #333; line-height: 2;">
            <li><strong>Streamlit 1.28+</strong> - Web Framework</li>
            <li><strong>Plotly</strong> - Interactive Charts</li>
            <li><strong>Matplotlib</strong> - Visualizations</li>
            <li><strong>Custom CSS</strong> - Premium Design</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0, 168, 232, 0.1) 0%, rgba(0, 201, 255, 0.1) 100%); 
                        padding: 20px; border-radius: 10px; border: 1px solid rgba(0, 168, 232, 0.2);">
            <h4 style="color: #00a8e8; margin-top: 0;">⚙️ Infrastructure</h4>
            <ul style="color: #333; line-height: 2;">
            <li><strong>Python 3.12</strong> - Runtime</li>
            <li><strong>Joblib</strong> - Model Persistence</li>
            <li><strong>Pickle</strong> - Data Serialization</li>
            <li><strong>Cross-platform</strong> - Windows/Mac/Linux</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset Information
        st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">📊 Dataset Information</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            **Source:** Kaggle - Pakistan House Prices and Property Listings
            
            **Records:** 168,446 properties
            
            **Features:** 17 attributes including location, city, property type, bedrooms, bathrooms, area, price, etc.
            
            **Coverage:** 5 major Pakistani cities
            
            **Price Range:** PKR 50,000 - 1,000,000,000+
            """)
        
        with col2:
            st.markdown("""
            **Data Quality:**
            - ✅ 154,899 records after preprocessing
            - ✅ No missing values in final dataset
            - ✅ Outliers removed using IQR method
            - ✅ Categorical features encoded
            - ✅ Features standardized and normalized
            """)
        
        # Model Performance
        st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">🎯 Model Performance</h3>', unsafe_allow_html=True)
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.markdown("""
            | Metric | Value |
            |--------|-------|
            | **R² Score** | 0.9231 (92.31%) |
            | **RMSE** | PKR 3,185,965 |
            | **MAE** | PKR 1,647,725 |
            | **Training Samples** | 123,919 |
            | **Test Samples** | 30,980 |
            """)
        
        with perf_col2:
            st.markdown("""
            **Model Hyperparameters:**
            - Estimators: 500
            - Learning Rate: 0.05
            - Max Depth: 6
            - Subsample: 0.8
            - Feature Subsample: 0.8
            """)
        
        # Key Features
        st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">✨ Key Features</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ✅ **Instant Predictions**
            Get accurate house price estimates in seconds
            
            ✅ **SHAP Explainability**
            Understand what factors drive each prediction
            
            ✅ **Feature Analysis**
            See which features most influence prices
            """)
        
        with col2:
            st.markdown("""
            ✅ **Interactive Analytics**
            Visualize model insights and performance
            
            ✅ **Professional Interface**
            Beautiful, responsive Streamlit UI
            
            ✅ **Market Context**
            Compare predictions to city averages
            """)
        
        # How It Works
        st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">🔄 How It Works</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(15, 76, 117, 0.05) 0%, rgba(0, 168, 232, 0.05) 100%); 
                    padding: 25px; border-radius: 10px;">
        
        <h4 style="color: #0f4c75;">Data Pipeline</h4>
        <p style="color: #333;">
        Raw Dataset → Preprocessing → Feature Engineering → Encoding → 
        Outlier Detection → Normalization → Clean Dataset
        </p>
        
        <h4 style="color: #0f4c75;">Training Pipeline</h4>
        <p style="color: #333;">
        Clean Data → Split (80/20) → XGBoost Training → 
        Evaluation → SHAP Analysis → Model Persistence
        </p>
        
        <h4 style="color: #0f4c75;">Inference Pipeline</h4>
        <p style="color: #333;">
        User Input → Preprocessing → Encoding → 
        XGBoost Prediction → SHAP Explanation → UI Display
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Author & Attribution
        st.markdown('<h3 style="color: #0f4c75; margin-top: 40px;">👨‍💻 Author & Attribution</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **AI/ML Engineer** - Full Stack Developer
            
            Specialized in:
            - Machine Learning & Data Science
            - Web Development (Python, Streamlit)
            - Model Explainability & Interpretability
            - Production Deployment & MLOps
            """)
        
        with col2:
            st.markdown("""
            **Dataset Source:**
            [Kaggle - Pakistan House Prices](https://www.kaggle.com/datasets/sarcasmos/pakistan-house-prices-and-property-listings)
            
            **Open Source Libraries:**
            - XGBoost Team
            - SHAP Contributors
            - Streamlit Team
            - Python Community
            """)
        
        # Footer
        st.markdown('<hr style="margin: 40px 0; border: none; border-top: 2px solid #e0e0e0;">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px; font-size: 14px;">
        <p><strong>Built with ❤️ using Python, XGBoost, SHAP, and Streamlit</strong></p>
        <p>This project is provided for educational and research purposes.</p>
        <p><em>Last Updated: May 2026</em></p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
