import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="Portfolio Risk Profile", layout="wide", page_icon="📊")

# --- LOAD DATA ---
data_path = Path(__file__).parent.parent.parent / "data" / "powerbi_credit_risk_predictions.csv"

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

# --- HEADER ---
st.markdown("<h2 style='color: #002060; margin-bottom: 0;'>Portfolio Risk Profile</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; margin-top: 5px;'>Analysis of risk concentration by Loan Term, DTI, FICO, and Purpose.</p>", unsafe_allow_html=True)
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
    st.metric("High-Risk Exposure", "0.28%") # From PBI dashboard
with col4:
    st.metric("Actual Default Rate", f"{actual_def_rate:.2f}%")

st.divider()

# --- ROW 1: Term Donut & DTI Band ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Total Exposure by Term")
    term_df = df.groupby('term', as_index=False)['loan_amnt'].sum()
    fig_term = px.pie(term_df, values='loan_amnt', names='term', hole=0.6, 
                      color_discrete_sequence=['#002060', '#1f77b4'])
    fig_term.update_traces(textinfo='percent+label', textposition='inside')
    fig_term.update_layout(margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_term, use_container_width=True)

with col2:
    st.subheader("Model Risk by DTI Band")
    # Create DTI Bands (using your labels from the screenshot)
    bins = [-10, 0, 10, 20, 30, 40, 50, 60, 100]
    labels = ['<0', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+']
    
    df_dti = df.copy()
    df_dti['dti_band'] = pd.cut(df_dti['dti'], bins=bins, labels=labels)
    
    dti_df = df_dti.groupby('dti_band', observed=True, as_index=False)['predicted_default_prob'].mean()
    
    fig_dti = px.bar(dti_df, x='dti_band', y='predicted_default_prob', color_discrete_sequence=['#002060'])
    fig_dti.update_layout(
        xaxis_title="DTI Band", 
        yaxis_title="Avg Predicted Risk", 
        yaxis_tickformat='.0%',
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_dti, use_container_width=True)

# --- ROW 2: FICO Combo Chart & Purpose ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("Model Risk by FICO Score Band")
    
    # Filter out the 0s and extreme outliers
    fico_df = df[(df['fico_range_low'] >= 600) & (df['fico_range_low'] <= 850)]
    
    fico_df = fico_df.groupby('fico_range_low', as_index=False).agg(
        Total_Loans=('is_default', 'count'),
        Predicted_Risk=('predicted_default_prob', 'mean')
    )
    
    fig_fico = go.Figure()
    # Bar for Total Loans
    fig_fico.add_trace(go.Bar(x=fico_df['fico_range_low'], y=fico_df['Total_Loans'], 
                              name='Total Loans', marker_color='#002060', yaxis='y1'))
    # Line for Predicted Risk
    fig_fico.add_trace(go.Scatter(x=fico_df['fico_range_low'], y=fico_df['Predicted_Risk'], 
                                  name='Avg Predicted Risk', mode='lines', line=dict(color='#1f77b4', width=3), yaxis='y2'))
    
    fig_fico.update_layout(
        yaxis=dict(title='Total Loans'),
        yaxis2=dict(title='Predicted Risk', overlaying='y', side='right', tickformat='.0%'),
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_fico, use_container_width=True)

with col4:
    st.subheader("Loan Exposure by Purpose")
    
    # Group and filter top 10 (removes junk text)
    purpose_df = df.groupby('purpose', as_index=False)['loan_amnt'].sum()
    purpose_df = purpose_df.sort_values(by='loan_amnt', ascending=False).head(10)
    purpose_df = purpose_df.sort_values(by='loan_amnt', ascending=True)
    
    fig_purpose = px.bar(purpose_df, x='loan_amnt', y='purpose', orientation='h', color_discrete_sequence=['#002060'])
    fig_purpose.update_layout(
        xaxis_title="Total Loan Amount", 
        yaxis_title="", 
        margin=dict(t=30, b=0, l=0, r=0),
        xaxis=dict(tickformat="$.2s") 
    )
    st.plotly_chart(fig_purpose, use_container_width=True)