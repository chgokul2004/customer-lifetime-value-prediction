########## Customer Lifetime Value Implementation #################################
import pandas as pd
import numpy as np
import datetime 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("--- Starting CLTV Calculation ---")

## ---------------- Data Loading -----------------------------
try:
    # Try reading with standard encoding
    df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("ERROR: 'OnlineRetail.csv' not found. Please download it from Kaggle and place it in this folder.")
    exit()
except Exception as e:
    print(f"ERROR: Could not read file. {e}")
    exit()

## ---------------- Preprocessing -----------------------------
# Clean Column Names (Strip spaces just in case)
df.columns = df.columns.str.strip()

# Check if 'CustomerID' exists (Some datasets spell it 'Customer ID')
if 'CustomerID' not in df.columns and 'Customer ID' in df.columns:
    df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)

# Drop missing IDs and create core fields
df = df.dropna(subset=['CustomerID'])
df['gross_sale'] = df['Quantity'] * df['UnitPrice']
df['reqdate'] = pd.to_datetime(df['InvoiceDate'])

# Rename to match your original script variables
df.rename(columns={
    'CustomerID': 'subscriberid',
    'InvoiceNo': 'invoiceid',
    'Country': 'area',
    'Description': 'description'
}, inplace=True)

# Ensure IDs are strings
df['subscriberid'] = df['subscriberid'].astype(int).astype(str)

# Filter out returns (negative sales)
data = df[df['gross_sale'] > 0].copy()
data = data.sort_values(by='reqdate')

print(f"Processed {len(data)} transactions.")

## ---------------- CLTV Logic -----------------------------

# Dynamic Area Filter: Pick the top area automatically to ensure data exists
top_area = data['area'].value_counts().idxmax()
print(f"Filtering for top area: {top_area}")
area_data = data[data['area'] == top_area]

# Grouping by Subscriber
final_data = data[['subscriberid', 'invoiceid', 'reqdate', 'gross_sale']].copy()

final_data_group = final_data.groupby('subscriberid').agg({
    'reqdate': lambda x: (x.max() - x.min()).days,
    'invoiceid': 'count',
    'gross_sale': 'sum'
}).reset_index()

final_data_group.columns = ['subscriberid', 'num_days', 'num_transactions', 'spent_money']

# 1. Average Order Value
final_data_group['avg_order_value'] = final_data_group['spent_money'] / final_data_group['num_transactions']

# 2. Purchase Frequency
purchase_frequency = final_data_group['num_transactions'].sum() / final_data_group.shape[0]

# 3. Repeat Rate & Churn
repeat_rate = final_data_group[final_data_group['num_transactions'] > 1].shape[0] / final_data_group.shape[0]
churn_rate = 1 - repeat_rate

# 4. Profit Margin (5%)
final_data_group['profit_margin'] = final_data_group['spent_money'] * 0.05

# 5. CLTV
# Note: Standard CLTV is (Values / Churn) * Margin%. 
# Your original code used: ((AOV * Freq) / Churn) * Total_Profit_Amount. 
# We will stick to your logic to keep consistency with your request.
final_data_group['CLV'] = (final_data_group['avg_order_value'] * purchase_frequency) / churn_rate
final_data_group['cust_lifetime_value'] = final_data_group['CLV'] * final_data_group['profit_margin']

print("CLTV Metrics Calculated.")
print(final_data_group[['subscriberid', 'cust_lifetime_value']].head())

## ---------------- Prediction Model -----------------------------
print("\n--- Starting Prediction Model ---")

# Prepare Time-Series Data
real_data = data.copy()
real_data['month_yr'] = real_data['reqdate'].dt.to_period('M')

# Pivot: Rows=Customer, Cols=Month, Values=Sales
sale = real_data.pivot_table(index='subscriberid', columns='month_yr', values='gross_sale', aggfunc='sum', fill_value=0)

# Calculate Target (CLV = Sum of all historical value in this context)
sale['CLV'] = sale.sum(axis=1)

# Feature Selection: Use last 6 months of data to predict CLV
# We ensure we have at least 6 months of data
if sale.shape[1] > 7:
    # Select features (Last 6 months before the CLV column)
    # The columns are [Month1, Month2, ..., MonthN, CLV]
    # We take Month(N-5) to Month(N)
    X = sale.iloc[:, -7:-1] 
    y = sale['CLV']

    # Remove inactive users (0 sales in last 6 months)
    active_mask = X.sum(axis=1) > 0
    X = X[active_mask]
    y = y[active_mask]
    
    # Feature Scaling note: Tree models don't need it, Linear models might benefit, 
    # but for simplicity we proceed as raw data.
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Define Models
    models = [
        LinearRegression(),
        Ridge(alpha=0.01),
        Lasso(),
        DecisionTreeRegressor(random_state=0),
        RandomForestRegressor(n_estimators=50, random_state=0),
        SVR(kernel='rbf')
    ]

    results = []

    for model in models:
        model_name = model.__class__.__name__
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = metrics.r2_score(y_test, pred)
            results.append((model_name, r2))
            print(f"{model_name}: R2 Score = {r2:.4f}")
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")

    # Find Best Model
    if results:
        best_model = max(results, key=lambda x: x[1])
        print(f"\nBest Performing Model: {best_model[0]} with R2: {best_model[1]:.4f}")
        
        # DataFrame Summary
        res_df = pd.DataFrame(results, columns=['Model', 'R2_Score'])
        print(res_df)
else:
    print("Not enough monthly data to build a 6-month lag model.")

print("--- Execution Complete ---")