

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/data/Processed/feature_engineered_sales.csv", parse_dates=['date'])
df = df.sort_values("date")


if 'store_name' in df.columns:
    store_mapping = dict(zip(df['store'], df['store_name']))
    df['store_name_full'] = df['store'].map(store_mapping)
else:
    df['store_name_full'] = "Store " + df['store'].astype(str)
    store_mapping = dict(zip(df['store'], df['store_name_full']))


df['month'] = df['date'].dt.to_period('M').astype(str)
df['day_name'] = df['date'].dt.day_name()
df['holiday_type'] = df['holiday_flag'].map({0: 'Non-Holiday', 1: 'Holiday'})

# Sidebar Filters
st.sidebar.title("🛏️ Filter Options")

store_options = sorted(df['store'].unique())
selected_stores = st.sidebar.multiselect("Select Store(s)", store_options, default=[store_options[0]],
                                         format_func=lambda x: store_mapping.get(x, f"Store {x}"))


date_range = st.sidebar.date_input("Select Date Range", [df['date'].min(), df['date'].max()])
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])


promo_filter = st.sidebar.radio("Promotion Week", options=["all", "promo", "no_promo"], index=0,
                                format_func=lambda x: {"all": "All Weeks", "promo": "Promotion Only", "no_promo": "No Promotion"}[x])


holiday_filter = st.sidebar.radio("Holiday Filter", options=["all", "holiday", "non_holiday"], index=0,
                                  format_func=lambda x: {"all": "All Days", "holiday": "Holiday Weeks Only", "non_holiday": "Non-Holiday Weeks"}[x])


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


st.title("📊 Walmart Store Sales Overview")
tab1, tab2, tab3 = st.tabs(["📈 Time-Series Overview", "📊 Exploratory Analysis", "🌡️ External Factors & Correlation"])


# Tab 1: Time-Series Overview

with tab1:

    st.subheader("📌 Sales Summary")
    total_sales = filtered_df['weekly_sales'].sum()
    avg_sales = filtered_df['weekly_sales'].mean()
    peak_date = filtered_df.loc[filtered_df['weekly_sales'].idxmax(), 'date']
    peak_value = filtered_df['weekly_sales'].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📉 Avg Weekly Sales", f"${avg_sales:,.0f}")
    col3.metric("📈 Peak Sales Day", f"{peak_date.date()}")

    if not filtered_df.empty:
        weekly = filtered_df.groupby(['store', 'date'])['weekly_sales'].sum().reset_index()
        weekly['store_name'] = weekly['store'].map(store_mapping)
        fig_weekly = px.area(weekly, x='date', y='weekly_sales', color='store_name',
                             title="📅 Weekly Sales Over Time (Area Chart)")
        fig_weekly.update_layout(title_x=0.5)
        st.plotly_chart(fig_weekly, use_container_width=True)

        monthly = filtered_df.groupby(['store', 'month'])['weekly_sales'].sum().reset_index()
        monthly['store_name'] = monthly['store'].map(store_mapping)
        fig_monthly = px.bar(monthly, x='month', y='weekly_sales', color='store_name', barmode='group',
                             title="📦 Monthly Sales by Store (Grouped Bar)")
        fig_monthly.update_layout(xaxis_tickangle=-45, title_x=0.5)
        st.plotly_chart(fig_monthly, use_container_width=True)

        rolling_df = []
        for store in selected_stores:
            df_store = filtered_df[filtered_df['store'] == store]
            weekly_store = df_store.groupby('date')['weekly_sales'].sum().reset_index()
            weekly_store['rolling_avg'] = weekly_store['weekly_sales'].rolling(window=13).mean()
            weekly_store['store'] = store
            weekly_store['store_name'] = store_mapping[store]
            rolling_df.append(weekly_store)
        rolling_all = pd.concat(rolling_df, ignore_index=True)
        fig_roll = px.line(rolling_all, x='date', y='rolling_avg', color='store_name',
                           title="🔁 13-Week Rolling Average (Smoothed)")
        fig_roll.update_layout(title_x=0.5)
        st.plotly_chart(fig_roll, use_container_width=True)

        st.subheader("🏪 Total Sales by Store")
        store_sales = (
            filtered_df.groupby('store')['weekly_sales']
            .sum()
            .reset_index()
            .sort_values(by='weekly_sales', ascending=False)
        )
        store_sales['store_name'] = store_sales['store'].map(store_mapping)
        fig_store_sales = px.bar(store_sales, x='store_name', y='weekly_sales', color='store_name',
                                 title="🏪 Total Sales by Store",
                                 labels={'store_name': 'Store', 'weekly_sales': 'Total Sales'})
        fig_store_sales.update_layout(showlegend=False, title_x=0.5)
        st.plotly_chart(fig_store_sales, use_container_width=True)

        st.subheader("⏱️ Time Series Decomposition (All Stores Combined)")
        ts = filtered_df.groupby('date')['weekly_sales'].sum().resample('W').sum()
        ts = ts.interpolate().fillna(method='bfill')
        result = seasonal_decompose(ts, model='additive', period=52)

        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        result.observed.plot(ax=axes[0], title='Observed')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonality')
        result.resid.plot(ax=axes[3], title='Residuals')
        st.pyplot(fig)
    else:
        st.warning("No data available for selected filters.")


# 📊 Tab 2: Exploratory Analysis

with tab2:
    st.subheader("📅 Sales by Day of Week")
    fig_day = px.box(df, x='day_name', y='weekly_sales', title='Sales by Day of Week')
    st.plotly_chart(fig_day, use_container_width=True)

    st.subheader("🎉 Holiday vs Non-Holiday Sales")
    fig_holiday = px.box(df, x='holiday_type', y='weekly_sales', title='Sales: Holiday vs Non-Holiday')
    st.plotly_chart(fig_holiday, use_container_width=True)

    st.subheader("📊 Weekly Sales Distribution")
    fig_dist = px.histogram(df, x='weekly_sales', nbins=50, title='Weekly Sales Distribution')
    st.plotly_chart(fig_dist, use_container_width=True)


# 🌡️ Tab 3: External Factors & Correlation

with tab3:
    st.subheader("⛽ Fuel Price vs Weekly Sales")
    fig_fuel = px.scatter(df, x='fuel_price', y='weekly_sales', trendline='ols')
    st.plotly_chart(fig_fuel, use_container_width=True)

    st.subheader("💼 Unemployment vs Weekly Sales")
    fig_unemp = px.scatter(df, x='unemployment', y='weekly_sales', trendline='ols')
    st.plotly_chart(fig_unemp, use_container_width=True)

    st.subheader("🌡️ Temperature vs Weekly Sales")
    fig_temp = px.scatter(df, x='temperature', y='weekly_sales', trendline='ols')
    st.plotly_chart(fig_temp, use_container_width=True)

    st.subheader("🔗 Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr.round(2),
        texttemplate="%{text}"
    ))
    fig_corr.update_layout(title='Correlation Heatmap')
    st.plotly_chart(fig_corr, use_container_width=True)
