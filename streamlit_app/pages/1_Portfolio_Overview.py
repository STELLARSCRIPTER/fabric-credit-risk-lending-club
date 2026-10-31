import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Portfolio Overview", layout="wide", page_icon="📊")

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

# Actually load the data and handle errors
try:
    df_raw = load_data()
except FileNotFoundError:
    st.error(f"Data file not found at: {data_path}")
    st.info("Please ensure the CSV is in the data/ folder and pushed to GitHub.")
    st.stop()

# --- SIDEBAR FILTER (Matches your Power BI Slicer) ---
st.sidebar.header("Filter Data")
home_options = df_raw['home_ownership'].unique().tolist()
selected_home = st.sidebar.multiselect("Home Ownership", home_options, default=home_options)

# Apply filter
df = df_raw[df_raw['home_ownership'].isin(selected_home)]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- HEADER ---
st.markdown("<h2 style='color: #002060; margin-bottom: 0;'>Portfolio Overview</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; margin-top: 5px;'>Portfolio composition and key credit-risk indicators.</p>", unsafe_allow_html=True)
st.divider()

# --- TOP KPIs ---
total_exposure = df['loan_amnt'].sum() / 1e9
weighted_avg_risk = df['predicted_default_prob'].mean() * 100
actual_def_rate = df['is_default'].mean() * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Exposure", f"${total_exposure:.2f}bn")
with col2:
    st.metric("Weighted Avg Risk", f"{weighted_avg_risk:.2f}%")
with col3:
    st.metric("High-Risk Exposure", "0.28%") # Static from PBI dashboard
with col4:
    st.metric("Actual Default Rate", f"{actual_def_rate:.2f}%")

st.divider()

# --- ROW 1: Calibration (Predicted vs Actual) & FICO ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Risk vs. Actual Risk")
    
    # Create bins for predicted risk (0.0 to 0.7 based on your PBI image)
    df_calib = df.copy()
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
    df_calib['risk_bin'] = pd.cut(df_calib['predicted_default_prob'], bins=bins, labels=labels)
    
    calib_df = df_calib.groupby('risk_bin', observed=True).agg(
        Total_Loans=('is_default', 'count'),
        Actual_Default_Rate=('is_default', 'mean')
    ).reset_index()
    
    fig_cal = go.Figure()
    # Bar for Loan Volume
    fig_cal.add_trace(go.Bar(x=calib_df['risk_bin'], y=calib_df['Total_Loans'], 
                          name='Total Loans', marker_color='#002060', yaxis='y1'))
    # Line for Actual Default Rate
    fig_cal.add_trace(go.Scatter(x=calib_df['risk_bin'], y=calib_df['Actual_Default_Rate']*100, 
                              name='Actual Default Rate (%)', mode='lines+markers', 
                              line=dict(color='#1f77b4', width=3), yaxis='y2'))
    
    fig_cal.update_layout(
        yaxis=dict(title='Total Loans'),
        yaxis2=dict(title='Actual Default Rate (%)', overlaying='y', side='right', range=[0, 70]),
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_cal, use_container_width=True)

with col2:
    st.subheader("Avg Predicted Risk by FICO Score")
    
    # 1. FILTER: Only keep valid FICO scores (600 to 850) to drop the 0s and outliers
    fico_df = df[(df['fico_range_low'] >= 600) & (df['fico_range_low'] <= 850)]
    
    # 2. Group by FICO score
    fico_df = fico_df.groupby('fico_range_low', as_index=False)['predicted_default_prob'].mean()
    
    # 3. Plot
    fig_fico = px.line(fico_df, x='fico_range_low', y='predicted_default_prob')
    fig_fico.update_traces(line=dict(color='#002060', width=3, dash='dot'), mode='lines+markers')
    
    # 4. Update layout to format the Y-axis as a percentage properly
    fig_fico.update_layout(
        xaxis_title="FICO Score",
        yaxis_title="Avg Predicted Risk", 
        yaxis_tickformat='.0%', # This makes it show 20% instead of 20.00000000%
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_fico, use_container_width=True)

# --- ROW 2: Purpose & Home Ownership ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("Loan Amount by Purpose")
    
    # Group and filter top 10 (removes junk text)
    purpose_df = df.groupby('purpose', as_index=False)['loan_amnt'].sum()
    purpose_df = purpose_df.sort_values(by='loan_amnt', ascending=False).head(10)
    purpose_df = purpose_df.sort_values(by='loan_amnt', ascending=True)
    
    fig_purpose = px.bar(purpose_df, x='loan_amnt', y='purpose', orientation='h', color_discrete_sequence=['#002060'])
    fig_purpose.update_layout(
        xaxis_title="Total Loan Amount", 
        yaxis_title="", 
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickformat="$.2s") 
    )
    st.plotly_chart(fig_purpose, use_container_width=True)

with col4:
    st.subheader("Borrowers by Home Ownership")
    
    # Note: This chart responds to the filter! If you deselect "RENT", the chart updates.
    home_df = df.groupby('home_ownership', as_index=False)['loan_amnt'].sum()
    fig_home = px.pie(home_df, values='loan_amnt', names='home_ownership', hole=0.6, 
                      color_discrete_sequence=px.colors.sequential.Blues_r)
    fig_home.update_traces(textposition='inside', textinfo='percent+label')
    fig_home.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_home, use_container_width=True)