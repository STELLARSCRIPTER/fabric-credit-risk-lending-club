import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="ML Predictions", layout="wide", page_icon="🤖")

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
    df = pd.read_csv(data_path)
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Data file not found at: {data_path}")
    st.stop()

# --- HEADER ---
st.markdown("<h2 style='color: #002060; margin-bottom: 0;'>ML Predictions</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; margin-top: 5px;'>Exploration of XGBoost model outputs and their relationship with actual outcomes.</p>", unsafe_allow_html=True)
st.divider()

# --- TOP KPIs ---
total_loans = len(df)
avg_pred_risk = df['predicted_default_prob'].mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Algorithm", "XGBoost Classifier")
with col2:
    st.metric("Total Loans Scored", f"{total_loans:,.0f}")
with col3:
    st.metric("Avg Predicted Risk", f"{avg_pred_risk:.2f}%")

st.divider()

# --- ROW 1: Time Series & Actual vs Predicted ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Actual Default Rate by Issue Date")
    df_time = df.dropna(subset=['issue_d']).copy()
    df_time['Issue_Year_Month'] = df_time['issue_d'].dt.to_period('M').astype(str)
    
    trend_df = df_time.groupby('Issue_Year_Month', as_index=False)['is_default'].mean()
    
    fig_trend = px.line(trend_df, x='Issue_Year_Month', y='is_default', color_discrete_sequence=['#002060'])
    fig_trend.update_layout(
        xaxis_title="", 
        yaxis_title="Actual Default Rate", 
        yaxis_tickformat='.0%',
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.subheader("Actual vs Predicted")
    df_calib = df.copy()
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
    df_calib['risk_bin'] = pd.cut(df_calib['predicted_default_prob'], bins=bins, labels=labels)
    
    calib_df = df_calib.groupby('risk_bin', observed=True).agg(
        Total_Loans=('is_default', 'count'),
        Actual_Default_Rate=('is_default', 'mean')
    ).reset_index()
    
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(x=calib_df['risk_bin'], y=calib_df['Total_Loans'], 
                          name='Total Loans', marker_color='#002060', yaxis='y1'))
    fig_cal.add_trace(go.Scatter(x=calib_df['risk_bin'], y=calib_df['Actual_Default_Rate']*100, 
                              name='Actual Default Rate', mode='lines+markers', 
                              line=dict(color='#1f77b4', width=3), yaxis='y2'))
    
    fig_cal.update_layout(
        yaxis=dict(title='Total Loans'),
        yaxis2=dict(title='Actual Default Rate', overlaying='y', side='right', range=[0, 70], tickformat='.0%'),
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_cal, use_container_width=True)

# --- ROW 2: Scatter Plot & Histogram ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("Interest Rate vs. Predicted Default Risk")
    
    df_scatter = df.sample(n=min(10000, len(df)), random_state=42)
    
    fig_scatter = px.scatter(
        df_scatter, 
        x='int_rate', 
        y='predicted_default_prob',
        color='verification_status',
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_scatter.update_layout(
        xaxis_title="Interest Rate", 
        yaxis_title="Predicted Default Prob",
        yaxis_tickformat='.0%',
        legend=dict(title="", orientation='h', y=-0.2),
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    st.subheader("Risk Distribution Histogram")
    
    verif_options = df['verification_status'].unique().tolist()
    selected_verif = st.multiselect("Filter by Verification Status", verif_options, default=verif_options)
    
    df_hist = df[df['verification_status'].isin(selected_verif)]
    
    fig_hist = px.histogram(
        df_hist, 
        x='predicted_default_prob', 
        nbins=30,
        color_discrete_sequence=['#002060']
    )
    fig_hist.update_layout(
        xaxis_title="Predicted Default Probability", 
        yaxis_title="Count of Loans",
        xaxis_tickformat='.0%',
        bargap=0.1,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_hist, use_container_width=True)