
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------
# 🧭 Dashboard Tabs
# ---------------------------
st.set_page_config(layout="wide", page_title="📈 Walmart Sales Forecast Dashboard")
st.title("📈 Walmart Sales Intelligence Platform")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Sales Overview", "🤖 Model Performance", "📈 Forecast Simulator", "🧠 Explainability", "📥 Export"])



# ---------------------------
# 📁 Load Data
# ---------------------------
df = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/data/processed/feature_engineered_sales.csv", parse_dates=['date'])
pred = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/outputs/predictions.csv", parse_dates=['date'])
resid = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/outputs/residuals.csv")
metrics = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/outputs/model_comparison.csv")



# Confirm store_name is in the dataset and use it
store_mapping = dict(zip(df['store'], df['store_name']))

# Merge predicted sales using store_name
df = df.merge(pred[['date', 'store', 'predicted_sales']], on=['date', 'store'], how='left')

# ---------------------------
# 🎛️ Sidebar Filters
# ---------------------------
st.sidebar.header("Filter Data")
store_list = sorted(df['store_name'].unique())
selected_stores = st.sidebar.multiselect("Select Stores", store_list, default=store_list[:3])
date_range = st.sidebar.date_input("Select Date Range", [df['date'].min(), df['date'].max()])




# Filter Data
filtered_df = df[df['store_name'].isin(selected_stores)]
filtered_df = filtered_df[(filtered_df['date'] >= pd.to_datetime(date_range[0])) & (filtered_df['date'] <= pd.to_datetime(date_range[1]))]

# Promo filter
promo_filter = st.sidebar.radio("Promotion Week", options=["all", "promo", "no_promo"],
                                format_func=lambda x: {"all": "All Weeks", "promo": "Promotion Only", "no_promo": "No Promotion"}[x])

# Holiday filter
holiday_filter = st.sidebar.radio("Holiday Filter", options=["all", "holiday", "non_holiday"],
                                  format_func=lambda x: {"all": "All Days", "holiday": "Holiday Weeks Only", "non_holiday": "Non-Holiday Weeks"}[x])

if promo_filter == "promo":
    filtered_df = filtered_df[filtered_df['promo_last_week'] == 1]
elif promo_filter == "no_promo":
    filtered_df = filtered_df[filtered_df['promo_last_week'] == 0]

if holiday_filter == "holiday":
    filtered_df = filtered_df[filtered_df['holiday_flag'] == 1]
elif holiday_filter == "non_holiday":
    filtered_df = filtered_df[filtered_df['holiday_flag'] == 0]

# ---------------------------
# 📊 Tab 1: Sales Overview
# ---------------------------
with tab1:
    st.subheader("📦 Key KPIs")

    if not filtered_df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"${filtered_df['weekly_sales'].sum():,.0f}")
        col2.metric("Avg Weekly Sales", f"${filtered_df['weekly_sales'].mean():,.0f}")
        peak_day = filtered_df.loc[filtered_df['weekly_sales'].idxmax(), 'date']
        col3.metric("Peak Sales Day", str(peak_day.date()))

    
        st.subheader("📅 Weekly Sales Trend")
        fig_line = px.line(filtered_df, x='date', y='weekly_sales', color='store_name', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("📦 Monthly Sales by Store")
        filtered_df['month'] = filtered_df['date'].dt.to_period('M').astype(str)
        month_sales = filtered_df.groupby(['store_name', 'month'])['weekly_sales'].sum().reset_index()
        fig_month = px.bar(month_sales, x='month', y='weekly_sales', color='store_name', barmode='group')
        st.plotly_chart(fig_month, use_container_width=True)

        st.subheader("🏪 Store Performance Ranking")
        top_stores = filtered_df.groupby('store_name')['weekly_sales'].sum().reset_index().sort_values(by='weekly_sales', ascending=False)
        fig_top = px.bar(top_stores, x='store_name', y='weekly_sales', color='store_name')
        st.plotly_chart(fig_top, use_container_width=True)

 

        st.subheader("🗺️ Sales by Store on Map")

        # Load store location data
        location_df = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/data/raw/store_locations.csv")

        # Aggregate sales for selected filters
        map_df = filtered_df.groupby('store_name', as_index=False)['weekly_sales'].sum()
        map_df = map_df.merge(location_df, on='store_name', how='left')

        # Sort by sales for better scaling
        map_df = map_df.sort_values(by="weekly_sales", ascending=False)

        # Format hover text
        map_df['hover_text'] = map_df['store_name'] + "<br>Total Sales: $" + map_df['weekly_sales'].round(0).astype(int).astype(str)

        # Plot Map
        fig_sales_map = px.scatter_map(
            map_df,
            lat="lat",
            lon="lon",
            size="weekly_sales",
            color="weekly_sales",
            hover_name="store_name",
            hover_data={"lat": False, "lon": False, "weekly_sales": ":,.0f"},
            text="store_name",
            size_max=40,
            zoom=3,
            color_continuous_scale="Blues",
            map_style="open-street-map",
            title="🧭 Total Sales by Store Location"
        )

        # Show map
        fig_sales_map.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_colorbar=dict(title="Sales"),
            title_x=0.5
        )
        st.plotly_chart(fig_sales_map, use_container_width=True)


        st.markdown("<br><br><br>", unsafe_allow_html=True)


        st.subheader("🚨 Inventory Alert Hotspots (High Sales Weeks)")

        # Highlight stores with unusually high sales
        weekly_threshold = filtered_df['weekly_sales'].quantile(0.95)  # Top 5% sales
        alert_df = filtered_df[filtered_df['weekly_sales'] > weekly_threshold].groupby('store_name')['weekly_sales'].count().reset_index()
        alert_df.columns = ['store_name', 'high_demand_weeks']
        alert_df = alert_df.merge(location_df, on='store_name', how='left')

        fig_alerts = px.scatter_map(
            alert_df,
            lat="lat",
            lon="lon",
            size="high_demand_weeks",
            color="high_demand_weeks",
            hover_name="store_name",
            size_max=40,
            zoom=3,
            map_style="open-street-map",
            title="Stores with Frequent High-Demand Weeks"
        )
        st.plotly_chart(fig_alerts, use_container_width=True)


    else:
        st.warning("⚠️ No data available for selected filters.")


   

    
   

# ---------------------------
# 🤖 Tab 2: Model Performance
# ---------------------------
with tab2:
    st.subheader("📋 Model Comparison Metrics")
    # Highlight the model with the lowest RMSE
    best_model = metrics.loc[metrics['RMSE'].idxmin(), 'Model']

    def highlight_best_model(row):
        return ['background-color: lightgreen' if row['Model'] == best_model else '' for _ in row]

    st.dataframe(
        metrics.style
        .format({'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'R² Score': '{:.4f}'})
        .apply(highlight_best_model, axis=1)
    )


    
    st.subheader("📉 Actual vs Predicted Sales")
    forecast_df = filtered_df.dropna(subset=['predicted_sales'])
    fig_forecast = px.line(forecast_df, x='date', y=['weekly_sales', 'predicted_sales'], color_discrete_sequence=['blue', 'orange'])
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("⚖️ Residuals vs Predicted")
    fig_resid = px.scatter(resid, x='predicted', y='residual', color='store_name', opacity=0.6)
    fig_resid.add_hline(y=0, line_dash='dot')
    st.plotly_chart(fig_resid, use_container_width=True)

# ---------------------------
# 📈 Tab 3: Forecast Simulator (Live with toggles)
# ---------------------------
from prophet import Prophet

with tab3:
    st.subheader("🔮 Future Sales Forecast Simulator")

    selected_store_name = st.selectbox("Select a Store for Forecasting", store_list)
    forecast_horizon = st.slider("Select Forecast Horizon (Weeks)", min_value=4, max_value=156, value=120, step=4)

    # Toggle promotion/holiday impact simulation
    simulate_promo = st.checkbox("Simulate Promotion Weeks?", value=False)
    simulate_holiday = st.checkbox("Simulate Holiday Weeks?", value=False)

    # Prepare data for Prophet
    store_df = df[df['store_name'] == selected_store_name][['date', 'weekly_sales']].rename(columns={'date': 'ds', 'weekly_sales': 'y'})
    store_df = store_df.groupby('ds').sum().reset_index()

    m = Prophet()
    m.fit(store_df)

    future = m.make_future_dataframe(periods=forecast_horizon, freq='W')
    forecast = m.predict(future)

    # Apply simulation multiplier
    multiplier = 1.0
    if simulate_promo:
        multiplier *= 1.10  # +10% uplift
    if simulate_holiday:
        multiplier *= 1.05  # +5% uplift

    forecast['yhat'] *= multiplier
    forecast['yhat_upper'] *= multiplier
    forecast['yhat_lower'] *= multiplier

    # Plot forecast
    fig_forecast = px.line()
    fig_forecast.add_scatter(x=store_df['ds'], y=store_df['y'], mode='lines+markers', name='Historical Sales')
    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast')
    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(dash='dot'), name='Upper Bound')
    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(dash='dot'), name='Lower Bound')
    fig_forecast.update_layout(title=f"{selected_store_name} Forecast - Next {forecast_horizon} Weeks",
                               xaxis_title="Date", yaxis_title="Sales", legend_title="Legend")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("### Forecast Table")
    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon)
    st.dataframe(forecast_table.rename(columns={
        'ds': 'Week',
        'yhat': 'Predicted Sales',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    }).style.format({
        'Predicted Sales': '{:,.2f}',
        'Lower Bound': '{:,.2f}',
        'Upper Bound': '{:,.2f}'
    }))




# ---------------------------
# 🧠 Tab 4: SHAP Explainability (Static Image)
# ---------------------------
with tab4:
    st.subheader("🧠 SHAP Explainability")

    st.markdown("### 🖼️ Interactive SHAP Summary")

    try:
        with open("C:/Users/Saisa/Downloads/Walmart_Sales/outputs/shap_summary_interactive.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    except FileNotFoundError:
        st.warning("SHAP summary file not found. Please generate shap_summary_interactive.html.")



# ---------------------------
# 📥 Tab 5: Export
# ---------------------------
with tab5:
    st.subheader("📤 Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="filtered_sales.csv", mime='text/csv')
