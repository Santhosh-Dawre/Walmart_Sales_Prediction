import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/data/Processed/feature_engineered_sales.csv", parse_dates=['date'])

# Create store name mapping
if 'store_name' in df.columns:
    store_mapping = dict(zip(df['store'], df['store_name']))
    df['store_name_full'] = df['store'].map(store_mapping)
else:
    df['store_name_full'] = "Store " + df['store'].astype(str)
    store_mapping = dict(zip(df['store'], df['store_name_full']))

# Add month column
df['month'] = df['date'].dt.to_period('M').astype(str)

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.title("🛏️ Filter Options")

store_options = sorted(df['store'].unique())
selected_stores = st.sidebar.multiselect("Select Store(s)", store_options, default=[store_options[0]],
                                         format_func=lambda x: store_mapping.get(x, f"Store {x}"))

# Date Range
date_range = st.sidebar.date_input("Select Date Range", [df['date'].min(), df['date'].max()])
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Promo filter
promo_filter = st.sidebar.radio("Promotion Week", options=["all", "promo", "no_promo"], index=0,
                                format_func=lambda x: {"all": "All Weeks", "promo": "Promotion Only", "no_promo": "No Promotion"}[x])

# Holiday filter
holiday_filter = st.sidebar.radio("Holiday Filter", options=["all", "holiday", "non_holiday"], index=0,
                                  format_func=lambda x: {"all": "All Days", "holiday": "Holiday Weeks Only", "non_holiday": "Non-Holiday Weeks"}[x])

# ---------------------------
# Filter Data
# ---------------------------
filtered_df = df[df['store'].isin(selected_stores)]
filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

if promo_filter == "promo":
    filtered_df = filtered_df[filtered_df['promo_last_week'] == 1]
elif promo_filter == "no_promo":
    filtered_df = filtered_df[filtered_df['promo_last_week'] == 0]

if holiday_filter == "holiday":
    filtered_df = filtered_df[filtered_df['holiday_flag'] == 1]
elif holiday_filter == "non_holiday":
    filtered_df = filtered_df[filtered_df['holiday_flag'] == 0]

# ---------------------------
# Charts
# ---------------------------
st.title("📊 Walmart Store Sales Overview")

if not filtered_df.empty:
    for store in selected_stores:
        store_name = store_mapping.get(store, f"Store {store}")
        st.subheader(f"📅 {store_name}")

        df_store = filtered_df[filtered_df['store'] == store]

        # Weekly Sales
        weekly = df_store.groupby('date')['weekly_sales'].sum().reset_index()
        fig_weekly = px.line(weekly, x='date', y='weekly_sales', title="Weekly Sales Over Time")
        fig_weekly.update_traces(mode='lines+markers')
        fig_weekly.update_layout(title_x=0.5)
        st.plotly_chart(fig_weekly, use_container_width=True)

        # Monthly Sales
        monthly = df_store.groupby('month')['weekly_sales'].sum().reset_index()
        fig_monthly = px.bar(monthly, x='month', y='weekly_sales', title="Monthly Sales")
        fig_monthly.update_layout(xaxis_tickangle=-45, title_x=0.5)
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Rolling Average
        weekly['rolling_avg'] = weekly['weekly_sales'].rolling(window=13).mean()
        fig_roll = px.line(weekly, x='date', y='rolling_avg', title="13-Week Rolling Average")
        fig_roll.update_traces(mode='lines+markers')
        fig_roll.update_layout(title_x=0.5)
        st.plotly_chart(fig_roll, use_container_width=True)

else:
    st.warning("No data available for selected filters.")
