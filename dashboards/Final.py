import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🚨 set_page_config must be the very first Streamlit call
st.set_page_config(layout="wide", page_title="Sales Forecast Dashboard")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/Saisa/Downloads/Walmart_Sales/data/processed/feature_engineered_sales.csv', parse_dates=['date'])
    model = joblib.load('C:/Users/Saisa/Downloads/Walmart_Sales/models/1_xgboost_sales_model.pkl')
    return df, model

df, model = load_data()

# Title
st.title("📊 Walmart Sales Forecast Dashboard")

# Sidebar filters
st.sidebar.header("🔍 Filters")

store_names = df['store_name'].unique()
selected_store_name = st.sidebar.selectbox("Select Store", store_names)

# Filter by selected store
selected_store = df[df['store_name'] == selected_store_name]['store'].iloc[0]

# Date range filter
date_range = st.sidebar.date_input("Select Date Range",
                                   [df['date'].min(), df['date'].max()])
selected_start, selected_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Filter data based on selections
filtered_df = df[(df['store'] == selected_store) &
                 (df['date'] >= selected_start) & (df['date'] <= selected_end)]

# Features for prediction
features = [col for col in df.columns
            if col not in ['date', 'weekly_sales', 'store', 'dept', 'store_name'] and df[col].dtype != 'object']

X_filtered = filtered_df[features]
y_true = filtered_df['weekly_sales']
y_pred = model.predict(X_filtered)

# Actual vs Predicted plot
st.subheader("📈 Actual vs Predicted Weekly Sales")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(filtered_df['date'], y_true.values, label='Actual', marker='o')
ax.plot(filtered_df['date'], y_pred, label='Predicted', marker='x')
ax.set_title(f"Store: {selected_store_name}")
ax.set_ylabel("Weekly Sales")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Evaluation Metrics
# Evaluation Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Manual RMSE
r2 = r2_score(y_true, y_pred)

st.markdown("### 📊 Model Evaluation Metrics")
st.write(f"**MAE:** {mae:,.2f}")
st.write(f"**RMSE:** {rmse:,.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Download button
result_df = filtered_df[['date', 'weekly_sales']].copy()
result_df['predicted_sales'] = y_pred
csv = result_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name='predicted_sales.csv',
    mime='text/csv'
)
