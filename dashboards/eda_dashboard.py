import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff

# Load feature-engineered data
df = pd.read_csv("C:/Users/Saisa/Downloads/Walmart_Sales/data/Processed/feature_engineered_sales.csv", parse_dates=['date'])

# Setup Dash app
app = dash.Dash(__name__)
app.title = "Walmart Sales EDA Dashboard"

# App Layout
app.layout = html.Div([
    html.H1("📊 Walmart Sales Interactive EDA", style={'textAlign': 'center'}),

    # KPI Summary
    html.Div(id='kpi_summary', style={
        'display': 'flex', 'justifyContent': 'space-around', 'padding': '10px',
        'borderBottom': '1px solid #ccc', 'marginBottom': '10px'
    }),

    # Filters
    html.Div([
        html.Label("Select Store:"),
        dcc.Dropdown(
            options=[{'label': f'Store {i}', 'value': i} for i in sorted(df['store'].unique())],
            id='store_selector',
            multi=True,
            value=[1]
        ),
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date_picker',
            start_date=df['date'].min(),
            end_date=df['date'].max()
        ),
        html.Label("Promotion Week:"),
        dcc.RadioItems(
            id='promo_filter',
            options=[
                {'label': 'All', 'value': 'all'},
                {'label': 'Promo Only', 'value': 'promo'},
                {'label': 'No Promo', 'value': 'no_promo'},
            ],
            value='all',
            inline=True
        )
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '20px'}),

    html.Hr(),

    # Tabs for Chart Groups
    dcc.Tabs([
        dcc.Tab(label='📊 Time Series', children=[
            dcc.Graph(id='sales_time_series'),
            dcc.Graph(id='sales_distribution'),
            dcc.Graph(id='rolling_sales_trend'),
            dcc.Graph(id='monthly_sales_trend'),
        ]),
        dcc.Tab(label='💰 Economic Factors', children=[
            dcc.Graph(id='fuel_vs_sales'),
            dcc.Graph(id='cpi_vs_sales'),
            dcc.Graph(id='temp_vs_sales'),
            dcc.Graph(id='unemployment_vs_sales'),
        ]),
        dcc.Tab(label='🏪 Store Analysis', children=[
            dcc.Graph(id='holiday_boxplot'),
            dcc.Graph(id='store_wise_bar'),
            dcc.Graph(id='promo_comparison')
        ]),
        dcc.Tab(label='🔁 Correlation Matrix', children=[
            dcc.Graph(id='correlation_heatmap')
        ])
    ])
])

# Callback
@app.callback(
    [
        Output('kpi_summary', 'children'),
        Output('sales_time_series', 'figure'),
        Output('sales_distribution', 'figure'),
        Output('holiday_boxplot', 'figure'),
        Output('fuel_vs_sales', 'figure'),
        Output('rolling_sales_trend', 'figure'),
        Output('monthly_sales_trend', 'figure'),
        Output('cpi_vs_sales', 'figure'),
        Output('temp_vs_sales', 'figure'),
        Output('unemployment_vs_sales', 'figure'),
        Output('store_wise_bar', 'figure'),
        Output('correlation_heatmap', 'figure'),
        Output('promo_comparison', 'figure')
    ],
    [
        Input('store_selector', 'value'),
        Input('date_picker', 'start_date'),
        Input('date_picker', 'end_date'),
        Input('promo_filter', 'value')
    ]
)
def update_all_charts(stores, start_date, end_date, promo_filter):
    dff = df[df['store'].isin(stores)]
    dff = dff[(dff['date'] >= start_date) & (dff['date'] <= end_date)]

    if promo_filter == 'promo':
        dff = dff[dff['promo_last_week'] == 1]
    elif promo_filter == 'no_promo':
        dff = dff[dff['promo_last_week'] == 0]

    # KPIs
    total_sales = dff['weekly_sales'].sum()
    avg_sales = dff['weekly_sales'].mean()
    num_stores = dff['store'].nunique()
    date_range = f"{dff['date'].min().date()} to {dff['date'].max().date()}"

    kpi_cards = [
        html.Div([
            html.H4("🧾 Total Sales"),
            html.P(f"${total_sales:,.0f}")
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'textAlign': 'center'}),

        html.Div([
            html.H4("📅 Date Range"),
            html.P(date_range)
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'textAlign': 'center'}),

        html.Div([
            html.H4("🏬 Stores Selected"),
            html.P(f"{num_stores}")
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'textAlign': 'center'}),

        html.Div([
            html.H4("📊 Avg Weekly Sales"),
            html.P(f"${avg_sales:,.0f}")
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'textAlign': 'center'}),
    ]

    # Charts
    time_series = px.line(dff.groupby('date')['weekly_sales'].sum().reset_index(),
                          x='date', y='weekly_sales', title="Weekly Sales Over Time")

    distribution = px.histogram(dff, x='weekly_sales', nbins=50,
                                title="Sales Distribution", marginal='box')

    boxplot = px.box(dff, x='holiday_flag', y='weekly_sales',
                     title="Sales: Holiday vs Non-Holiday")

    fuel_plot = px.scatter(dff, x='fuel_price', y='weekly_sales',
                           trendline='ols', title="Fuel Price vs Sales")

    roll_fig = px.line(dff, x='date', y='sales_roll_13',
                       title="13-week Rolling Sales", color='store')

    dff['month_str'] = dff['date'].dt.to_period('M').astype(str)
    monthly_trend = px.bar(dff.groupby('month_str')['weekly_sales'].sum().reset_index(),
                           x='month_str', y='weekly_sales', title="Monthly Sales Trend")

    cpi_plot = px.scatter(dff, x='cpi', y='weekly_sales',
                          trendline='ols', title="CPI vs Weekly Sales")

    temp_plot = px.scatter(dff, x='temperature', y='weekly_sales',
                           trendline='ols', title="Temperature vs Weekly Sales")

    unemp_plot = px.scatter(dff, x='unemployment', y='weekly_sales',
                            trendline='ols', title="Unemployment vs Weekly Sales")

    store_bar = px.bar(dff.groupby('store')['weekly_sales'].sum().reset_index(),
                       x='store', y='weekly_sales', title="Total Sales by Store")

    # Correlation Heatmap
    corr_df = dff[['weekly_sales', 'temperature', 'fuel_price', 'cpi', 'unemployment']].corr()
    z = corr_df.values.round(2)
    x = list(corr_df.columns)
    y = list(corr_df.index)

    corr_fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='RdBu',
                                           showscale=True, zmin=-1, zmax=1)
    corr_fig.update_layout(title="Correlation Matrix")

    # Promo vs No Promo
    promo_df = dff.copy()
    promo_df['promo_week'] = promo_df['promo_last_week'].map({1: 'Promo', 0: 'No Promo'})
    promo_comparison = px.box(promo_df, x='promo_week', y='weekly_sales',
                              title="Weekly Sales: Promo vs No Promo",
                              labels={'promo_week': 'Promotion Status'})

    return kpi_cards, time_series, distribution, boxplot, fuel_plot, roll_fig, monthly_trend, cpi_plot, temp_plot, unemp_plot, store_bar, corr_fig, promo_comparison

# Run App
if __name__ == '__main__':
    app.run(debug=True)
