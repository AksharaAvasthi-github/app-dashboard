import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

st.write("🚀 Streamlit app started successfully")

# Load datasets
orders = pd.read_csv("cost_breakdown.csv")
vehicles = pd.read_csv("customer_feedback.csv")
warehouses = pd.read_csv("delivery_performance.csv")
delivery_perf = pd.read_csv("orders.csv")
traffic_weather = pd.read_csv("routes_distance.csv")
costs = pd.read_csv("vehicle_fleet.csv")
carbon = pd.read_csv("warehouse_inventory.csv")

# Show dataset previews
for name, df in zip(
    ["Orders", "Vehicles", "Warehouses", "Delivery Performance", "Traffic & Weather", "Costs", "Carbon/Inventory"],
    [delivery_perf, vehicles, warehouses, delivery_perf, traffic_weather, costs, carbon]
):
    st.write(f"### {name} Dataset")
    st.dataframe(df.head())

# Merge data
df = (delivery_perf
      .merge(warehouses, on='Order_ID', how='left')
      .merge(traffic_weather[['Order_ID','Distance_KM','Traffic_Delay_Minutes','Weather_Impact']], on='Order_ID', how='left')
      .merge(vehicles[['Order_ID','Rating','Issue_Category']], on='Order_ID', how='left')
     )

# Handle missing data
numeric_cols = ['Promised_Delivery_Days', 'Actual_Delivery_Days', 'Customer_Rating', 'Delivery_Cost_INR',
                'Distance_KM', 'Traffic_Delay_Minutes']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

categorical_cols = ['Carrier', 'Quality_Issue', 'Special_Handling']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Map delivery and weather impact
priority_map = {'On-Time': 3, 'Slightly-Delayed': 2, 'Severely-Delayed': 1}
df['priority_score'] = df['Delivery_Status'].map(priority_map)

weather_map = {'None': 0, 'Light_Rain': 1, 'Heavy_Rain': 2, 'Storm': 3, 'Fog': 1, 'Snow': 2}
df['Weather_Impact'] = df['Weather_Impact'].fillna('None')
df['Weather_Impact_Score'] = df['Weather_Impact'].map(weather_map)

df['delayed'] = (df['Actual_Delivery_Days'] > df['Promised_Delivery_Days']).astype(int)

# Model setup
feature_cols = ['priority_score', 'Distance_KM', 'Traffic_Delay_Minutes', 'Weather_Impact_Score',
                'Customer_Rating', 'Delivery_Cost_INR']

feature_cols += [col for col in df.columns if 'Product_Category_' in col or 'Carrier_' in col]
X = df[feature_cols]
y = df['delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))

df['delay_prob'] = model.predict_proba(X)[:,1]

# Order weight calc
weights_map = {
    'Product_Category_Electronics': 5,
    'Product_Category_Fashion': 2,
    'Product_Category_Food & Beverage': 10,
    'Product_Category_Healthcare': 8,
    'Product_Category_Home Goods': 7,
    'Product_Category_Industrial': 50
}

df['Order_Weight_KG'] = sum(df[col] * weight for col, weight in weights_map.items() if col in df.columns)

available_vehicles = costs[costs['Status'] == 'Available'].copy()

# Vehicle assignment
def assign_vehicle(order):
    candidates = available_vehicles[available_vehicles['Capacity_KG'] >= order['Order_Weight_KG']]
    if candidates.empty:
        return None
    candidates['Score'] = (
        (1 / (candidates['Fuel_Efficiency_KM_per_L'] + 0.01)) +
        candidates['CO2_Emissions_Kg_per_KM'] +
        (candidates['Current_Location'] != order['Origin']).astype(int) * 5
    )
    best_vehicle = candidates.loc[candidates['Score'].idxmin()]
    return best_vehicle['Vehicle_ID']

df['Assigned_Vehicle'] = df.apply(assign_vehicle, axis=1)

st.dataframe(df[['Order_ID', 'Order_Weight_KG', 'Origin', 'Assigned_Vehicle']].head(10))

orders_df = df
vehicles_df = costs

# CO2 merge
orders_with_co2 = orders_df.merge(
    vehicles_df[['Vehicle_ID', 'CO2_Emissions_Kg_per_KM']],
    left_on='Assigned_Vehicle',
    right_on='Vehicle_ID',
    how='left'
)
st.success(f"✅ orders_with_co2 created: {orders_with_co2.shape}")

orders_with_co2['Total_CO2_Kg'] = orders_with_co2['CO2_Emissions_Kg_per_KM'] * orders_with_co2['Distance_KM']

# Merge deliveries
orders_full = orders_with_co2.merge(
    delivery_perf[['Order_ID', 'Order_Value_INR', 'Origin', 'Destination', 'Priority']],
    on='Order_ID', how='left'
)
st.success(f"✅ after merging delivery_perf: {orders_full.shape}")

orders_full = orders_full.merge(
    traffic_weather[['Order_ID', 'Fuel_Consumption_L', 'Toll_Charges_INR', 'Traffic_Delay_Minutes', 'Weather_Impact']],
    on='Order_ID', how='left'
)

FUEL_PRICE = 100
orders_full['Fuel_Cost_INR'] = orders_full['Fuel_Consumption_L'] * FUEL_PRICE
orders_full['Total_Delivery_Cost_INR'] = orders_full['Fuel_Cost_INR'] + orders_full['Toll_Charges_INR']

orders_full = orders_full.merge(
    warehouses[['Order_ID', 'Delivery_Cost_INR']],
    on='Order_ID',
    how='left'
)
st.success(f"✅ after merging warehouses: {orders_full.shape}")

orders_full.rename(columns={'Delivery_Cost_INR': 'Total_Delivery_Cost_INR'}, inplace=True)

orders_full = orders_full.merge(
    orders_with_co2[['Order_ID', 'Distance_KM', 'Total_CO2_Kg']],
    on='Order_ID',
    how='left'
)

st.write("🔹 orders_with_co2 columns:", list(orders_with_co2.columns))
st.write("🔹 orders_full columns BEFORE SCALING:", list(orders_full.columns))

# --- FIXED SCALING SECTION ---
scaler = MinMaxScaler()

if 'Total_Delivery_Cost_INR' in orders_full.columns:
    cost_col = 'Total_Delivery_Cost_INR'
elif 'Delivery_Cost_INR' in orders_full.columns:
    cost_col = 'Delivery_Cost_INR'
elif 'Delivery_Cost_INR_x' in orders_full.columns:
    cost_col = 'Delivery_Cost_INR_x'
else:
    raise KeyError("No delivery cost column found in orders_full")

cols_to_normalize = []
for base_col in ['Distance_KM', 'Total_CO2_Kg']:
    if base_col in orders_full.columns:
        cols_to_normalize.append(base_col)
    elif f"{base_col}_y" in orders_full.columns:
        cols_to_normalize.append(f"{base_col}_y")

if cost_col in orders_full.columns:
    cols_to_normalize.append(cost_col)

if cols_to_normalize:
    orders_full[cols_to_normalize] = scaler.fit_transform(orders_full[cols_to_normalize])
    st.success(f"✅ Scaled columns: {cols_to_normalize}")
else:
    st.warning("⚠️ No matching columns found to scale.")

# Handle missing safely
distance = orders_full['Distance_KM'] if 'Distance_KM' in orders_full.columns else (
    orders_full['Distance_KM_y'] if 'Distance_KM_y' in orders_full.columns else 0)
co2 = orders_full['Total_CO2_Kg'] if 'Total_CO2_Kg' in orders_full.columns else (
    orders_full['Total_CO2_Kg_y'] if 'Total_CO2_Kg_y' in orders_full.columns else 0)
cost = orders_full[cost_col] if cost_col in orders_full.columns else 0

orders_full['Delivery_Efficiency'] = 1 - (
    distance * 0.4 +
    co2 * 0.3 +
    cost * 0.3
)

orders_full['Delivery_Efficiency'] = MinMaxScaler().fit_transform(
    orders_full[['Delivery_Efficiency']]
)

orders_full = orders_full.merge(
    delivery_perf[['Order_ID', 'Priority']],
    on='Order_ID',
    how='left'
)

priority_map = {'Express': 3, 'Standard': 2, 'Economy': 1}
orders_full['Priority_Score'] = orders_full['Priority'].map(priority_map)

# Normalize numeric columns (also handle _y)
numeric_cols = []
for base_col in ['Distance_KM', 'Total_CO2_Kg', 'Delivery_Cost_INR']:
    if base_col in orders_full.columns:
        numeric_cols.append(base_col)
    elif f"{base_col}_y" in orders_full.columns:
        numeric_cols.append(f"{base_col}_y")
numeric_cols.append('Priority_Score')

orders_full[numeric_cols] = scaler.fit_transform(orders_full[numeric_cols])

weights = {
    'Priority_Score': 0.4,
    'Distance_KM': 0.2,
    'Total_CO2_Kg': 0.2,
    'Delivery_Cost_INR': 0.2
}

orders_full['Assignment_Score'] = (
    orders_full['Priority_Score'] * weights['Priority_Score'] -
    distance * weights['Distance_KM'] -
    co2 * weights['Total_CO2_Kg'] -
    cost * weights['Delivery_Cost_INR']
)

orders_full = orders_full.sort_values(by='Assignment_Score', ascending=False)

# Vehicle assignment final
vehicles_available = available_vehicles.copy()
orders_full['Assigned_Vehicle'] = None

for idx, order in orders_full.iterrows():
    if len(vehicles_available) == 0:
        break
    vehicle_id = vehicles_available.iloc[0]['Vehicle_ID']
    orders_full.at[idx, 'Assigned_Vehicle'] = vehicle_id
    vehicles_available = vehicles_available.iloc[1:]

st.dataframe(orders_full[['Order_ID', 'Assigned_Vehicle', 'Assignment_Score']].head(10))

orders_full = orders_full.merge(
    vehicles_df[['Vehicle_ID', 'CO2_Emissions_Kg_per_KM']],
    left_on='Assigned_Vehicle',
    right_on='Vehicle_ID',
    how='left'
)

co2_col = None
for possible_col in ['CO2_Emissions_Kg_per_KM', 'CO2_Emissions_Kg_per_KM_y', 'CO2_Emissions_Kg_per_KM_x']:
    if possible_col in orders_full.columns:
        co2_col = possible_col
        break

if co2_col is not None:
    distance_col = 'Distance_KM_y' if 'Distance_KM_y' in orders_full.columns else 'Distance_KM'
    orders_full['Total_CO2_Kg'] = orders_full[distance_col] * orders_full[co2_col]
    st.dataframe(
        orders_full[['Order_ID', 'Assigned_Vehicle', distance_col, co2_col, 'Total_CO2_Kg']].head()
    )
else:
    st.warning("⚠️ CO2 emissions column not found in merged data — skipping CO2 calculation.")
