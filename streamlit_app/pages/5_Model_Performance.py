import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Performance", layout="wide", page_icon="📈")

# --- ROBUST DATA PATH SETUP ---
# --- ROBUST DATA PATH SETUP ---
# Get the absolute path of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up two directories to reach the root of the repo (fabric-credit-risk-lending-club)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

data_path = os.path.join(ROOT_DIR, "data", "powerbi_credit_risk_predictions.csv")

# --- LOAD DATA FUNCTION ---
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Data file not found at: {data_path}")
    st.stop()

# --- HEADER ---
st.markdown("<h2 style='color: #002060; margin-bottom: 0;'>Model Performance & Validation</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; margin-top: 5px;'>Evaluation metrics, threshold analysis, and model limitations.</p>", unsafe_allow_html=True)
st.divider()

# --- TOP KPIs (Metrics) ---
# Hardcoded from your Power BI dashboard for exact matching
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Precision", "0.54")
with col2:
    st.metric("Recall", "0.05")
with col3:
    st.metric("F1-Score", "0.10")

st.divider()

# --- ROW 1: Confusion Matrix & Limitations ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Confusion Matrix (Threshold = 50%)")
    
    # Custom CSS to make the confusion matrix boxes look good
    st.markdown("""
    <style>
    .cm-box { padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px; }
    .cm-title { font-size: 14px; font-weight: bold; margin-bottom: 5px; }
    .cm-value { font-size: 28px; font-weight: bold; color: #002060; }
    .tp { background-color: #e6f0fa; border: 1px solid #b3d4f5; }
    .fn { background-color: #fff3cd; border: 1px solid #ffeeba; }
    .tn { background-color: #d4edda; border: 1px solid #c3e6cb; }
    .fp { background-color: #f8d7da; border: 1px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)
    
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown('<div class="cm-box tp"><div class="cm-title">True Positives (Caught Defaults)</div><div class="cm-value">2,896</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="cm-box tn"><div class="cm-title">True Negatives (Correctly Cleared)</div><div class="cm-value">212,935</div></div>', unsafe_allow_html=True)
    with cm_col2:
        st.markdown('<div class="cm-box fn"><div class="cm-title">False Negatives (Missed Defaults)</div><div class="cm-value">51,079</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="cm-box fp"><div class="cm-title">False Positives (False Alarms)</div><div class="cm-value">2,443</div></div>', unsafe_allow_html=True)

with col2:
    st.subheader("Model Limitations & Governance")
    st.info("""
    **Evaluation Period:** Trained on historical Lending Club data (2007–2018). Macroeconomic shifts (like the 2008 crisis) heavily influence default rates.
    
    **Threshold Choice:** The 50% threshold maximizes Precision but results in a high False Negative rate (missed defaults). A lower threshold (e.g., 20%) could be used to catch more defaults at the cost of more False Alarms.
    
    **Interpretability:** XGBoost is a complex ensemble model. It should be used to *support* credit policies, not replace human judgment.
    """)

st.divider()

# --- ROW 2: AUC Gauge & Scatter Plot ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("Model AUC")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 0.70,
        number = {'font': {'size': 50, 'color': '#002060'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0.5, 1.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#002060"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0.5, 0.7], 'color': '#f8d7da'},
                {'range': [0.7, 0.8], 'color': '#fff3cd'},
                {'range': [0.8, 1.0], 'color': '#d4edda'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7031
            }
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(t=30, b=0, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col4:
    st.subheader("Default Risk by Interest Rate")
    
    # Sample data for performance
    df_scatter = df.sample(n=min(10000, len(df)), random_state=42)
    
    fig_scatter = px.scatter(
        df_scatter, x='int_rate', y='predicted_default_prob', 
        color_discrete_sequence=['#002060'], opacity=0.5
    )
    fig_scatter.update_layout(
        xaxis_title="Interest Rate", 
        yaxis_title="Predicted Default Prob",
        yaxis_tickformat='.0%',
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- ROW 3: Predicted Risk Distribution ---
st.divider()
st.subheader("Predicted Risk Distribution")
fig_hist = px.histogram(df, x='predicted_default_prob', nbins=50, color_discrete_sequence=['#002060'])
fig_hist.update_layout(
    xaxis_title="Predicted Default Probability", 
    yaxis_title="Count of Loans",
    xaxis_tickformat='.0%',
    bargap=0.1,
    margin=dict(t=30, b=0, l=0, r=0)
)
st.plotly_chart(fig_hist, use_container_width=True)