import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import plotly.figure_factory as ff

# Title
st.title("🧪 Walmart Sales Data - EDA Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/data/Processed/feature_engineered_sales.csv", parse_dates=['date'])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("📅 Filters")
store_id = st.sidebar.selectbox("Select Store", options=["All"] + sorted(df['store'].unique().tolist()))

if store_id != "All":
    df = df[df['store'] == int(store_id)]

# 1. Time Series Trend
st.subheader("📈 Weekly Sales Over Time")
fig1 = px.line(df, x='date', y='weekly_sales', title='Weekly Sales Over Time')
st.plotly_chart(fig1)

# 2. 4-Week Rolling Average
st.subheader("📉 4-Week Rolling Average")
df['rolling_avg'] = df['weekly_sales'].rolling(window=4).mean()
fig2 = px.line(df, x='date', y='rolling_avg', title='4-Week Rolling Average Sales')
st.plotly_chart(fig2)

# 3. Total Sales by Store
st.subheader("🏪 Total Sales by Store")
store_sales = df.groupby('store')['weekly_sales'].sum().reset_index()
fig3 = px.bar(store_sales.sort_values('weekly_sales', ascending=False), x='store', y='weekly_sales')
st.plotly_chart(fig3)

# 4. Monthly Sales Trend
st.subheader("🗓 Monthly Sales Trend")
df['month'] = df['date'].dt.month
monthly_sales = df.groupby('month')['weekly_sales'].sum().reset_index()
fig4 = px.bar(monthly_sales, x='month', y='weekly_sales', title='Sales by Month')
st.plotly_chart(fig4)

# 5. Holiday vs Non-Holiday Sales
st.subheader("🎉 Sales During Holiday vs Non-Holiday")
fig5 = px.box(df, x='holiday_flag', y='weekly_sales', labels={'holiday_flag': 'Holiday Week'})
st.plotly_chart(fig5)

# 6. Scatter: Sales vs Temperature
st.subheader("🌡️ Sales vs Temperature")
fig6 = px.scatter(df, x='temperature', y='weekly_sales', trendline='ols')
st.plotly_chart(fig6)

# 7. Correlation Heatmap
st.subheader("🔍 Correlation Heatmap")
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr().round(2)
fig7 = ff.create_annotated_heatmap(z=corr_matrix.values, x=list(corr_matrix.columns), y=list(corr_matrix.columns), annotation_text=corr_matrix.values.astype(str), colorscale='Viridis')
st.plotly_chart(fig7)

# 8. Outlier Detection
st.subheader("🚨 Outlier Detection")
df['z_score'] = zscore(df['weekly_sales'])
df['outlier'] = df['z_score'].abs() > 3
fig8 = px.scatter(df, x='date', y='weekly_sales', color='outlier', title='Outliers Highlighted')
st.plotly_chart(fig8)
