# Walmart Sales Forecasting & Decision Support Dashboard
This project uses machine learning and time-series forecasting to predict future sales across Walmart stores in the US. It translates complex model outputs into actionable business insights through an interactive, user-friendly Streamlit dashboard.



## Project Objective
To build a data-driven forecasting system that helps retail planners and decision-makers:

- Predict future sales with high accuracy
- Simulate promotion or holiday scenarios
- Identify underperforming stores 
- Optimise business strategies using visual analytic



## Key Features:
- Sales Forecasting Models: Linear Regression, Random Forest, XGBoost, and Facebook Prophet  
- Model Performance Evaluation: MAE, RMSE, R² metrics, residual analysis  
- Interactive Filters: Store, date range, promotion, holiday  
- Geospatial Visualisation: Sales maps and inventory hotspot alerts  
- Explainability: SHAP summary for feature impact  
- Forecast Simulator : Date-based Prophet forecasts with upper/lower bounds
- What-if Simulation: Toggle promotions or holidays to simulate impact  
- Data Export: Download filtered results as CSV  
  


## Project Structure
Walmart_Sales/
├── data/ # Raw and processed datasets
│ ├── raw/
│ └── processed/
├── notebooks/ # Jupyter notebooks for Data Processing, Feature Engineering, EDA, modeling, explainability
├── models/ # Saved machine learning models
├── outputs/ # Metrics, plots, SHAP explainability files
├── dashboard/ # Streamlit dashboard (Eda_Dashboard & final_dashboard.py)
└── requirements.txt # Python dependencies



## How to Run
1. Clone repo - https://github.com/Santhosh-Dawre/Walmart_Sales_Prediction/
2. Install dependencies with `pip install -r requirements.txt`
3. Run analysis notebooks
4. Launch dashboard with `streamlit run dashboard/final_dashboard.py`



## Output
- Forecasted sales charts
- Scenario simulations (e.g., promotion, holidays)



## Dataset:
Source: [Walmart Sales Dataset on Kaggle](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)  
Includes historical weekly sales data, holidays, and economic indicators across 45 Walmart stores.



## Dashboard Tabs Overview
- Tab 1 (Sales Overview):  Trends, filters, store rankings, sales maps, and alerts 
- Tab 2 (Model Performance): Compare MAE, RMSE, R²; view residuals and predictions
- Tab 3 (Forecast Simulator): Prophet Based Forecasts with Scenario toggles
- Tab 4 (Explainability):  SHAP feature importance via interactive HTML
- Tab 5 (Export): Download filtered data for external use 



## Business Use Cases
- Plan promotions based on projected uplift
- Detect underperforming regions
- Understand what drives sales through SHAP insights


## Author
Leela Sai Santhosh Dawre  
MSc Data Science, Newcastle University  
Supervisor: Dr. Lei Shi  
Module: CSC8639 (Individual Project and Dissertation)

