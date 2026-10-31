import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="High-Risk Watchlist", layout="wide", page_icon="🚨")

# --- ROBUST DATA PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if 'pages' in BASE_DIR:
    ROOT_DIR = os.path.dirname(BASE_DIR)
else:
    ROOT_DIR = BASE_DIR

data_path = os.path.join(ROOT_DIR, "data", "powerbi_credit_risk_predictions.csv")

# --- LOAD DATA FUNCTION ---
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

# Filter to only high-risk loans (matching your Power BI threshold of >60%)
watchlist_df = df[df['predicted_default_prob'] > 0.60].copy()

# --- HEADER ---
st.markdown("<h2 style='color: #002060; margin-bottom: 0;'>High-Risk Watchlist</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; margin-top: 5px;'>Actionable list of loans flagged by the XGBoost model (>60% risk).</p>", unsafe_allow_html=True)
st.divider()

# --- TOP KPIs ---
total_flagged = len(watchlist_df)
portfolio_share = (total_flagged / len(df)) * 100
avg_risk_score = watchlist_df['predicted_default_prob'].mean() * 100
actual_def_rate = watchlist_df['is_default'].mean() * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Flagged High-Risk Loans", f"{total_flagged:,.0f}")
with col2:
    st.metric("High-Risk Portfolio Share", f"{portfolio_share:.2f}%")
with col3:
    st.metric("Average Risk Score (Flagged)", f"{avg_risk_score:.2f}%")
with col4:
    st.metric("Actual Default Rate (Flagged)", f"{actual_def_rate:.2f}%")

st.divider()

# --- ROW 1: Income Verification & Volume Trend ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("High-Risk Loans by Income Verification")
    verif_df = watchlist_df.groupby('verification_status', as_index=False)['loan_amnt'].count()
    verif_df.columns = ['verification_status', 'count']
    
    fig_verif = px.bar(verif_df, x='verification_status', y='count', color_discrete_sequence=['#002060'])
    fig_verif.update_layout(xaxis_title="", yaxis_title="Number of Loans", margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_verif, use_container_width=True)

with col2:
    st.subheader("High-Risk Loan Volume Trend (by Year)")
    watchlist_df['Issue_Year'] = watchlist_df['issue_d'].dt.year
    
    trend_df = watchlist_df.groupby('Issue_Year', as_index=False)['loan_amnt'].count()
    trend_df.columns = ['Issue_Year', 'count']
    
    fig_trend = px.line(trend_df, x='Issue_Year', y='count', color_discrete_sequence=['#002060'])
    fig_trend.update_layout(xaxis_title="", yaxis_title="Count of Loans", margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# --- ROW 2: Geographic Concentration & Primary Drivers ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("Geographic Concentration of High-Risk Loans")
    
    state_df = watchlist_df.groupby('addr_state', as_index=False)['loan_amnt'].count()
    state_df.columns = ['addr_state', 'count']
    state_df = state_df.sort_values(by='count', ascending=False).head(20)
    state_df = state_df.sort_values(by='count', ascending=True)
    
    fig_state = px.bar(state_df, x='count', y='addr_state', orientation='h', color_discrete_sequence=['#002060'])
    fig_state.update_layout(xaxis_title="Number of Loans", yaxis_title="", margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_state, use_container_width=True)

with col4:
    st.subheader("Primary Drivers of High-Risk Loans (by Purpose)")
    
    purpose_df = watchlist_df.groupby('purpose', as_index=False)['loan_amnt'].count()
    purpose_df.columns = ['purpose', 'count']
    purpose_df = purpose_df.sort_values(by='count', ascending=False).head(10)
    purpose_df = purpose_df.sort_values(by='count', ascending=True)
    
    fig_purpose = px.bar(purpose_df, x='count', y='purpose', orientation='h', color_discrete_sequence=['#002060'])
    fig_purpose.update_layout(xaxis_title="Number of Loans", yaxis_title="", margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_purpose, use_container_width=True)