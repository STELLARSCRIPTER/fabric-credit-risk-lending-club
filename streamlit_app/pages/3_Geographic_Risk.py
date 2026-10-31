import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="Geographic Risk", layout="wide", page_icon="🗺️")

# --- LOAD DATA ---
data_path = Path(__file__).parent.parent.parent / "data" / "powerbi_credit_risk_predictions.csv"

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

# --- HEADER ---
st.markdown("<h2 style='color: #002060; margin-bottom: 0;'>Geographic Risk Profile</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; margin-top: 5px;'>State-level loan exposure and default hotspots across the United States.</p>", unsafe_allow_html=True)
st.divider()

# --- TOP KPIs ---
total_states = df['addr_state'].nunique()
total_exposure = df['loan_amnt'].sum() / 1e9
actual_def_rate = df['is_default'].mean() * 100

# Find state with highest exposure
state_exposure = df.groupby('addr_state')['loan_amnt'].sum()
highest_exposure_state = state_exposure.idxmax()

# Find state with highest default rate (min 100 loans to avoid 1-loan outliers)
state_default = df.groupby('addr_state').agg(
    Total_Loans=('is_default', 'count'),
    Default_Rate=('is_default', 'mean')
)
state_default = state_default[state_default['Total_Loans'] > 100] 
highest_risk_state = state_default['Default_Rate'].idxmax()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total States", f"{total_states}")
with col2:
    st.metric("Total Exposure", f"${total_exposure:.2f}bn")
with col3:
    st.metric("Highest-Risk State", f"{highest_risk_state}")
with col4:
    st.metric("Highest-Exposure State", f"{highest_exposure_state}")
with col5:
    st.metric("Actual Default Rate", f"{actual_def_rate:.2f}%")

st.divider()

# --- ROW 1: Term & Map ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Default Rate by Loan Term")
    term_df = df.groupby('term', as_index=False)['is_default'].mean()
    
    fig_term = px.bar(term_df, x='term', y='is_default', color_discrete_sequence=['#002060'])
    fig_term.update_layout(
        xaxis_title="", 
        yaxis_title="Default Rate", 
        yaxis_tickformat='.0%',
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_term, use_container_width=True)

with col2:
    st.subheader("Default Rate by State")
    # Aggregate by state
    map_df = df.groupby('addr_state', as_index=False)['is_default'].mean()
    
    # Use Plotly's built-in US state map
    fig_map = px.choropleth(
        map_df, 
        locations='addr_state', 
        locationmode="USA-states", 
        color='is_default', 
        scope="usa",
        color_continuous_scale="Reds", # Matches your red Power BI map
        labels={'is_default': 'Default Rate'}
    )
    fig_map.update_layout(
        coloraxis_colorbar=dict(title="Default Rate", tickformat='.0%'),
        margin=dict(l=0, r=0, t=10, b=0),
        geo=dict(bgcolor='rgba(0,0,0,0)') # Transparent background
    )
    st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# --- ROW 2: Top States by Exposure & Risk ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("Top States by Loan Exposure")
    # Sort and get top 20 for a cleaner chart
    exp_df = state_exposure.reset_index().sort_values(by='loan_amnt', ascending=False).head(20)
    
    fig_exp = px.bar(exp_df, x='addr_state', y='loan_amnt', color_discrete_sequence=['#002060'])
    fig_exp.update_layout(
        xaxis_title="", 
        yaxis_title="Total Loan Amount", 
        margin=dict(t=30, b=0, l=0, r=0),
        yaxis=dict(tickformat="$.2s") 
    )
    st.plotly_chart(fig_exp, use_container_width=True)

with col4:
    st.subheader("Highest Default Rates by State")
    # Sort and get top 20
    def_df = state_default.reset_index().sort_values(by='Default_Rate', ascending=False).head(20)
    # Reverse so highest is at the top of the horizontal bar chart
    def_df = def_df.sort_values(by='Default_Rate', ascending=True)
    
    fig_def = px.bar(def_df, x='Default_Rate', y='addr_state', orientation='h', color_discrete_sequence=['#002060'])
    fig_def.update_layout(
        xaxis_title="Default Rate", 
        yaxis_title="", 
        xaxis_tickformat='.0%',
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_def, use_container_width=True)