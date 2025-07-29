import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Page Config (MUST BE FIRST)
# ---------------------------
st.set_page_config(layout="wide", page_title="Walmart Sales Forecast Dashboard")

# ---------------------------
# Load Data and Model
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/Saisa/Downloads/Walmart_Sales/data/processed/feature_engineered_sales.csv', parse_dates=['date'])
    model = joblib.load('C:/Users/Saisa/Downloads/Walmart_Sales/models/xgboost_sales_model.pkl')
    return df, model

df, model = load_data()

# ---------------------------
# Walmart Header with Logo
# ---------------------------
st.markdown("""
    <div style='display: flex; align-items: center;'>
        <img src='C:/Users/Saisa/Downloads/Walmart_Sales/1.png' width='100'/>
        <h1 style='margin-left: 20px;'>Walmart Sales Forecast Dashboard</h1>
    </div>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("\U0001F50D Filter Options")
store_mapping = dict(zip(df['store'], df['store_name']))
store_options = sorted(df['store'].unique())
selected_store = st.sidebar.selectbox("Select Store", store_options, format_func=lambda x: store_mapping[x])

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['date'].min(), df['date'].max()]
)
selected_start, selected_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# ---------------------------
# Data Filtering
# ---------------------------
filtered_df = df[(df['store'] == selected_store) &
                 (df['date'] >= selected_start) & (df['date'] <= selected_end)]

features = [col for col in df.columns if col not in ['date', 'weekly_sales', 'store', 'dept'] and df[col].dtype != 'object']
X_filtered = filtered_df[features]
y_true = filtered_df['weekly_sales']
y_pred = model.predict(X_filtered)

# ---------------------------
# Metrics Table
# ---------------------------
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

st.markdown("""
<style>
.metric-table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 10px;
    background-color: #f8f9fa;
    border-radius: 10px;
    overflow: hidden;
}}
.metric-table th, .metric-table td {{
    border: 1px solid #dee2e6;
    padding: 12px 16px;
    text-align: center;
}}
.metric-table th {{
    background-color: #004c91;
    color: white;
    font-weight: bold;
}}
.metric-table td {{
    font-size: 18px;
}}
</style>
<table class='metric-table'>
    <tr><th>MAE</th><th>RMSE</th><th>R² Score</th></tr>
    <tr><td>{mae:,.2f}</td><td>{rmse:,.2f}</td><td>{r2:.4f}</td></tr>
</table>
""".format(mae=mae, rmse=rmse, r2=r2), unsafe_allow_html=True)

# ---------------------------
# Actual vs Predicted Plot
# ---------------------------
st.subheader("\U0001F4C8 Actual vs Predicted Sales")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(filtered_df['date'], y_true.values, label='Actual', marker='o', color='navy')
ax.plot(filtered_df['date'], y_pred, label='Predicted', marker='x', linestyle='--', color='orange')
ax.set_title("Actual vs Predicted Sales", fontsize=16, fontweight='bold')
ax.set_ylabel("Weekly Sales")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)
