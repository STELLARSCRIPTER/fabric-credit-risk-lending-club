import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Executive Overview | Credit Risk", layout="wide", page_icon="📊")

# --- ROBUST DATA PATH SETUP ---
# Get the directory where the script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the root of the repo)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
# If the script is inside the 'pages' folder, go up one more level
if os.path.basename(CURRENT_DIR) == 'pages':
    ROOT_DIR = os.path.dirname(ROOT_DIR)

data_path = os.path.join(ROOT_DIR, "data", "powerbi_credit_risk_predictions.csv")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Data file not found at: {data_path}")
    st.stop()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .kpi-card {
        background-color: #f8f9fa; padding: 15px; border-radius: 10px;
        border-left: 5px solid #002060; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        height: 100%;
    }
    .kpi-value { font-size: 26px; font-weight: bold; color: #002060; }
    .kpi-label { font-size: 14px; color: #555; }
    .guide-box {
        background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER & INTERACTIVE SLIDER ---
col_header, col_slider = st.columns([2, 1])

with col_header:
    st.markdown("<h2 style='color: #002060; margin-bottom: 0;'>Executive Overview</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; font-style: italic; margin-top: 5px;'>Enterprise Credit Risk & Default Prediction Platform</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight: bold; margin-top: 0;'>Data Pipeline: PySpark (Bronze → Silver → Gold) | Model: XGBoost (AUC: 0.7031)</p>", unsafe_allow_html=True)

with col_slider:
    st.write("") 
    st.write("") 
    threshold = st.slider("Adjust High-Risk Prediction Threshold (%)", min_value=30, max_value=90, value=60, step=5)
    
    # Logic for dynamic KPIs
    high_risk_df = df[df['predicted_default_prob'] >= (threshold/100)]
    high_risk_exposure = (high_risk_df['loan_amnt'].sum() / df['loan_amnt'].sum()) * 100
    high_risk_actual_default = high_risk_df['is_default'].mean() * 100 if len(high_risk_df) > 0 else 0

st.divider()

# --- TOP KPIs ---
total_exposure = df['loan_amnt'].sum() / 1e9
actual_def_rate = df['is_default'].mean() * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Portfolio Exposure</div><div class="kpi-value">${total_exposure:.2f}bn</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Portfolio Default Rate</div><div class="kpi-value">{actual_def_rate:.2f}%</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">High-Risk Exposure Share</div><div class="kpi-value">{high_risk_exposure:.2f}%</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Actual Default Rate (Flagged)</div><div class="kpi-value">{high_risk_actual_default:.2f}%</div></div>', unsafe_allow_html=True)

st.write("") 

# --- ROW 1: Guide & Interest Rate Risk ---
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.markdown("""
    <div class="guide-box">
        <h3 style='margin-top:0;'>Dashboard Navigation Guide</h3>
        <ul style='line-height: 1.6; font-size: 15px;'>
            <li><b>Overview:</b> Executive KPIs and the Calibration Curve.</li>
            <li><b>Portfolio Risk Profile:</b> Risk by FICO, DTI, and Term.</li>
            <li><b>Geographic Risk:</b> State-level exposure and hotspots.</li>
            <li><b>Model Performance:</b> AUC, Precision/Recall, Confusion Matrix.</li>
            <li><b>High-Risk Watchlist:</b> Actionable list of flagged loans.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    st.markdown("""
    <div class="guide-box">
        <h3 style='margin-top:0;'>Key Influencers</h3>
        <ul style='line-height: 1.6; font-size: 14px;'>
            <li>Interest Rate is the strongest predictor.</li>
            <li>DTI above 21.19 significantly increases risk.</li>
            <li>FICO scores below 670 are highly correlated with defaults.</li>
            <li>Renters exhibit higher default rates than Mortgage holders.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("Default Risk by Interest Rate Band")
    bins = [0, 8.19, 11.39, 13.23, 19.53, 100]
    labels = ['8.19 or less', '8.19 - 11.39', '11.39 - 13.23', '13.23 - 19.53', 'more than 19.53']
    
    # Create a copy to avoid SettingWithCopy warnings
    df_int = df.copy()
    df_int['int_rate_band'] = pd.cut(df_int['int_rate'], bins=bins, labels=labels)
    int_rate_df = df_int.groupby('int_rate_band', observed=True, as_index=False)['is_default'].mean()
    
    fig_int = px.bar(int_rate_df, x='int_rate_band', y='is_default', color_discrete_sequence=['#002060'])
    fig_int.add_hline(y=actual_def_rate/100, line_dash="dot", line_color="green", annotation_text=f"Avg: {actual_def_rate:.2f}%")
    fig_int.update_layout(xaxis_title="", yaxis_title="Average Default Rate", yaxis_tickformat='.0%', margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_int, use_container_width=True)

st.divider()

# --- ROW 2: FICO & Calibration ---
col_fico, col_calib = st.columns(2)

with col_fico:
    st.subheader("Risk Decreases as FICO Score Increases")
    fico_df = df[(df['fico_range_low'] >= 600) & (df['fico_range_low'] <= 850)]
    fico_df = fico_df.groupby('fico_range_low', as_index=False).agg(Actual=('is_default', 'mean'), Predicted=('predicted_default_prob', 'mean'))
    fig_fico = px.line(fico_df, x='fico_range_low', y=['Actual', 'Predicted'], color_discrete_sequence=['#002060', '#1f77b4'])
    fig_fico.update_layout(yaxis_tickformat='.0%', legend=dict(x=0, y=1.1, orientation='h'), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_fico, use_container_width=True)

with col_calib:
    st.subheader("Model Calibration (Predicted vs Actual)")
    df_calib = df.copy()
    df_calib['risk_bin'] = pd.cut(df_calib['predicted_default_prob'], bins=np.arange(0, 1.1, 0.1), right=False)
    calib_df = df_calib.groupby('risk_bin', observed=True).agg(Loans=('is_default', 'count'), Actual=('is_default', 'mean')).reset_index()
    calib_df['risk_bin_str'] = calib_df['risk_bin'].astype(str)
    
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(x=calib_df['risk_bin_str'], y=calib_df['Loans'], name='Total Loans', marker_color='#002060', yaxis='y1'))
    fig_cal.add_trace(go.Scatter(x=calib_df['risk_bin_str'], y=calib_df['Actual']*100, name='Actual Default Rate (%)', mode='lines+markers', line=dict(color='#1f77b4', width=3), yaxis='y2'))
    fig_cal.update_layout(yaxis=dict(title='Total Loans'), yaxis2=dict(title='Actual Default Rate (%)', overlaying='y', side='right'), legend=dict(x=0, y=1.1, orientation='h'), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_cal, use_container_width=True)

st.divider()

# --- ROW 3: Demographics & Purpose ---
col_home, col_purpose = st.columns(2)

with col_home:
    st.subheader("Borrowers by Home Ownership")
    # Filtering out tiny categories (like ANY, NONE) to keep the donut clean
    home_df = df.groupby('home_ownership', as_index=False)['loan_amnt'].sum()
    home_df = home_df[home_df['loan_amnt'] > (df['loan_amnt'].sum() * 0.01)] # Keep only >1% share
    
    fig_home = px.pie(home_df, values='loan_amnt', names='home_ownership', hole=0.6, color_discrete_sequence=px.colors.sequential.Blues_r)
    fig_home.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_home, use_container_width=True)

with col_purpose:
    st.subheader("Loan Amount by Purpose")
    
    # 1. Group and sum
    purpose_df = df.groupby('purpose', as_index=False)['loan_amnt'].sum()
    
    # 2. FILTER: Drop junk text and tiny categories (keep only the top 10)
    purpose_df = purpose_df.sort_values(by='loan_amnt', ascending=False).head(10)
    
    # 3. Sort ascending for the horizontal bar chart
    purpose_df = purpose_df.sort_values(by='loan_amnt', ascending=True)
    
    fig_purpose = px.bar(purpose_df, x='loan_amnt', y='purpose', orientation='h', color_discrete_sequence=['#002060'])
    fig_purpose.update_layout(
        xaxis_title="Total Loan Amount", 
        yaxis_title="", 
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickformat="$.2s") # Formats to Millions/Billions cleanly
    )
    st.plotly_chart(fig_purpose, use_container_width=True)