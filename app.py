# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# ---------- Page config ----------
st.set_page_config(
    page_title="Smart Delivery Optimization",
    page_icon="🚚",
    layout="wide"
)

# ---------- Small theme polish ----------
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1115; color: #e6eef8; }
    .stSidebar { background-color: #0b0c0f; }
    .css-1d391kg { color: #e6eef8; } /* headers */
    .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helper functions ----------
def get_col(df, base_name):
    """Return the actual column name in df for base_name, checking variants like base_name, base_name_x, base_name_y."""
    if base_name in df.columns:
        return base_name
    if f"{base_name}_x" in df.columns:
        return f"{base_name}_x"
    if f"{base_name}_y" in df.columns:
        return f"{base_name}_y"
    return None

def safe_median_fill(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

def safe_fill_unknown(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

def try_train_model(X, y):
    """Train a RandomForest if features exist and return model or None."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, classification_report(y_test, y_pred, zero_division=0)
    except Exception:
        return None, None

# ---------- Sidebar (navigation + filters) ----------
st.sidebar.title("🚚 Delivery Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Data", "Visualizations", "Insights", "Export"])

st.sidebar.markdown("---")
st.sidebar.header("Filters")

# We'll set filter defaults after loading data

# ---------- Load data (relative paths) ----------
@st.cache_data(ttl=60*10)
def load_data():
    orders = pd.read_csv("cost_breakdown.csv")
    vehicles = pd.read_csv("customer_feedback.csv")
    warehouses = pd.read_csv("delivery_performance.csv")
    delivery_perf = pd.read_csv("orders.csv")
    traffic_weather = pd.read_csv("routes_distance.csv")
    costs = pd.read_csv("vehicle_fleet.csv")
    carbon = pd.read_csv("warehouse_inventory.csv")
    return {
        "orders": orders,
        "vehicles": vehicles,
        "warehouses": warehouses,
        "delivery_perf": delivery_perf,
        "traffic_weather": traffic_weather,
        "costs": costs,
        "carbon": carbon
    }

data = load_data()
orders = data["orders"]
vehicles = data["vehicles"]
warehouses = data["warehouses"]
delivery_perf = data["delivery_perf"]
traffic_weather = data["traffic_weather"]
costs = data["costs"]
carbon = data["carbon"]

# ---------- Quick preview area (for developers) ----------
if page == "Data":
    st.header("📁 Loaded datasets")
    left, right = st.columns(2)
    with left:
        st.subheader("delivery_perf (orders.csv)")
        st.dataframe(delivery_perf.head(8))
        st.subheader("warehouses (delivery_performance.csv)")
        st.dataframe(warehouses.head(6))
        st.subheader("traffic_weather (routes_distance.csv)")
        st.dataframe(traffic_weather.head(6))
    with right:
        st.subheader("vehicles (customer_feedback.csv)")
        st.dataframe(vehicles.head(6))
        st.subheader("vehicle fleet (vehicle_fleet.csv)")
        st.dataframe(costs.head(6))
        st.subheader("warehouse inventory")
        st.dataframe(carbon.head(4))

# ---------- Prepare and merge main dataframe (defensive) ----------
# Start from delivery_perf and progressively merge
main_df = delivery_perf.copy()

# Merge warehouses (if they have Order_ID)
if "Order_ID" in warehouses.columns:
    main_df = main_df.merge(warehouses, on="Order_ID", how="left")

# Merge traffic_weather selected cols defensively
tw_cols = [c for c in ["Order_ID", "Distance_KM", "Traffic_Delay_Minutes", "Weather_Impact", "Fuel_Consumption_L", "Toll_Charges_INR"] if c in traffic_weather.columns]
if tw_cols:
    main_df = main_df.merge(traffic_weather[tw_cols], on="Order_ID", how="left")

# Merge vehicles ratings
v_cols = [c for c in ["Order_ID", "Rating", "Issue_Category"] if c in vehicles.columns]
if v_cols:
    main_df = main_df.merge(vehicles[v_cols], on="Order_ID", how="left")

# Fill missing sensibly
safe_median_fill(main_df, ['Promised_Delivery_Days', 'Actual_Delivery_Days', 'Customer_Rating', 'Delivery_Cost_INR', 'Distance_KM', 'Traffic_Delay_Minutes'])
safe_fill_unknown(main_df, ['Carrier', 'Quality_Issue', 'Special_Handling'])
main_df['Delivery_Status'] = main_df['Delivery_Status'].fillna('On-Time') if 'Delivery_Status' in main_df.columns else None

# priority_score mapping if exists
if 'Delivery_Status' in main_df.columns:
    priority_map_small = {'On-Time': 3, 'Slightly-Delayed': 2, 'Severely-Delayed': 1}
    main_df['priority_score'] = main_df['Delivery_Status'].map(priority_map_small)

# weather impact mapping
if 'Weather_Impact' in main_df.columns:
    weather_map = {'None': 0, 'Light_Rain': 1, 'Heavy_Rain': 2, 'Storm': 3, 'Fog': 1, 'Snow': 2}
    main_df['Weather_Impact'] = main_df['Weather_Impact'].fillna('None')
    main_df['Weather_Impact_Score'] = main_df['Weather_Impact'].map(weather_map)

# delayed flag
if set(['Actual_Delivery_Days','Promised_Delivery_Days']).issubset(main_df.columns):
    main_df['delayed'] = (main_df['Actual_Delivery_Days'] > main_df['Promised_Delivery_Days']).astype(int)
else:
    main_df['delayed'] = 0

# Features for ML (defensive)
feature_cols = [c for c in ['priority_score', 'Distance_KM', 'Traffic_Delay_Minutes', 'Weather_Impact_Score','Customer_Rating', 'Delivery_Cost_INR'] if c in main_df.columns]
feature_cols += [col for col in main_df.columns if 'Product_Category_' in col or 'Carrier_' in col]
if feature_cols and 'delayed' in main_df.columns:
    X = main_df[feature_cols]
    y = main_df['delayed']
    model, report = try_train_model(X, y)
else:
    model, report = None, None

# Add delay probability (if model exists)
if model is not None and feature_cols:
    try:
        main_df['delay_prob'] = model.predict_proba(main_df[feature_cols])[:, 1]
    except Exception:
        main_df['delay_prob'] = 0
else:
    main_df['delay_prob'] = 0

# Order weight calculation
weights_map = {
    'Product_Category_Electronics': 5,
    'Product_Category_Fashion': 2,
    'Product_Category_Food & Beverage': 10,
    'Product_Category_Healthcare': 8,
    'Product_Category_Home Goods': 7,
    'Product_Category_Industrial': 50
}
main_df['Order_Weight_KG'] = 0
for col, w in weights_map.items():
    if col in main_df.columns:
        main_df['Order_Weight_KG'] += main_df[col] * w

# available vehicles from costs
available_vehicles = costs[costs['Status'] == 'Available'].copy() if 'Status' in costs.columns else costs.copy()
vehicles_df = costs.copy()

# assign_vehicle as in original logic
def assign_vehicle_row(row):
    if 'Order_Weight_KG' not in row:
        return None
    candidates = available_vehicles[available_vehicles['Capacity_KG'] >= row['Order_Weight_KG']] if 'Capacity_KG' in available_vehicles.columns else available_vehicles
    if candidates.empty:
        return None
    # compute score defensively
    fe_name = 'Fuel_Efficiency_KM_per_L'
    co2_name = 'CO2_Emissions_Kg_per_KM'
    loc_name = 'Current_Location'
    score = (1 / (candidates[fe_name] + 0.01)).fillna(9999) if fe_name in candidates.columns else 0
    score = score + candidates[co2_name].fillna(0) if co2_name in candidates.columns else score
    if loc_name in candidates.columns and 'Origin' in row:
        score = score + (candidates[loc_name] != row['Origin']).astype(int) * 5
    candidates = candidates.copy()
    candidates['Score'] = score
    try:
        best = candidates.loc[candidates['Score'].idxmin()]
        return best['Vehicle_ID'] if 'Vehicle_ID' in best else None
    except Exception:
        return None

main_df['Assigned_Vehicle'] = main_df.apply(assign_vehicle_row, axis=1)

# ---------------- CO2 + cost calculations into orders_full ----------------
orders_with_co2 = main_df.merge(
    vehicles_df[['Vehicle_ID', 'CO2_Emissions_Kg_per_KM']] if 'Vehicle_ID' in vehicles_df.columns and 'CO2_Emissions_Kg_per_KM' in vehicles_df.columns else vehicles_df,
    left_on='Assigned_Vehicle',
    right_on='Vehicle_ID',
    how='left'
)

# compute Total_CO2_Kg if possible (may use Distance_KM variants)
distance_col_for_orders_with = get_col(orders_with_co2, "Distance_KM")
co2_col_for_orders_with = get_col(orders_with_co2, "CO2_Emissions_Kg_per_KM")
if distance_col_for_orders_with and co2_col_for_orders_with:
    orders_with_co2['Total_CO2_Kg'] = orders_with_co2[distance_col_for_orders_with].fillna(0) * orders_with_co2[co2_col_for_orders_with].fillna(0)
else:
    orders_with_co2['Total_CO2_Kg'] = 0

orders_full = orders_with_co2.merge(
    delivery_perf[[c for c in ["Order_ID", "Order_Value_INR", "Origin", "Destination", "Priority"] if c in delivery_perf.columns]],
    on='Order_ID', how='left'
)

orders_full = orders_full.merge(
    traffic_weather[[c for c in ["Order_ID", "Fuel_Consumption_L", "Toll_Charges_INR", "Traffic_Delay_Minutes", "Weather_Impact"] if c in traffic_weather.columns]],
    on='Order_ID', how='left'
)

# fuel cost, total cost
FUEL_PRICE = 100
if 'Fuel_Consumption_L' in orders_full.columns:
    orders_full['Fuel_Cost_INR'] = orders_full['Fuel_Consumption_L'].fillna(0) * FUEL_PRICE
else:
    orders_full['Fuel_Cost_INR'] = 0
if 'Toll_Charges_INR' in orders_full.columns:
    orders_full['Toll_Charges_INR'] = orders_full['Toll_Charges_INR'].fillna(0)
else:
    orders_full['Toll_Charges_INR'] = 0

orders_full['Total_Delivery_Cost_INR'] = orders_full['Fuel_Cost_INR'] + orders_full['Toll_Charges_INR']

# merge warehouse delivery cost if present
if 'Delivery_Cost_INR' in warehouses.columns:
    orders_full = orders_full.merge(warehouses[['Order_ID', 'Delivery_Cost_INR']], on='Order_ID', how='left')

# rename if duplicated
if 'Delivery_Cost_INR_y' in orders_full.columns and 'Total_Delivery_Cost_INR' not in orders_full.columns:
    orders_full['Total_Delivery_Cost_INR'] = orders_full['Delivery_Cost_INR_y']

# add Distance_KM and Total_CO2_Kg from orders_with_co2 if missing
if get_col(orders_full, "Distance_KM") is None and get_col(orders_with_co2, "Distance_KM"):
    orders_full = orders_full.merge(orders_with_co2[['Order_ID', get_col(orders_with_co2, "Distance_KM")]], on='Order_ID', how='left')
if get_col(orders_full, "Total_CO2_Kg") is None and 'Total_CO2_Kg' in orders_with_co2.columns:
    orders_full = orders_full.merge(orders_with_co2[['Order_ID', 'Total_CO2_Kg']], on='Order_ID', how='left')

# ---------- SCALING & DELIVERY EFFICIENCY ----------
scaler = MinMaxScaler()

# determine cost column
if 'Total_Delivery_Cost_INR' in orders_full.columns:
    cost_col = 'Total_Delivery_Cost_INR'
elif 'Delivery_Cost_INR' in orders_full.columns:
    cost_col = 'Delivery_Cost_INR'
elif 'Delivery_Cost_INR_x' in orders_full.columns:
    cost_col = 'Delivery_Cost_INR_x'
elif 'Delivery_Cost_INR_y' in orders_full.columns:
    cost_col = 'Delivery_Cost_INR_y'
else:
    cost_col = None

# prepare list of columns to normalize with variants
cols_to_normalize = []
for base in ['Distance_KM', 'Total_CO2_Kg']:
    c = get_col(orders_full, base)
    if c:
        cols_to_normalize.append(c)
if cost_col:
    cols_to_normalize.append(cost_col)

# scale if possible
if cols_to_normalize:
    try:
        orders_full[cols_to_normalize] = scaler.fit_transform(orders_full[cols_to_normalize].fillna(0))
    except Exception:
        # fallback: scale per-column if multi-col scaling fails
        for c in cols_to_normalize:
            try:
                orders_full[[c]] = MinMaxScaler().fit_transform(orders_full[[c]].fillna(0))
            except Exception:
                pass

# compute Delivery_Efficiency using whichever columns exist
d_col = get_col(orders_full, "Distance_KM")
co2_col = get_col(orders_full, "Total_CO2_Kg")
cost_col = cost_col if cost_col in orders_full.columns else None

distance_series = orders_full[d_col] if d_col in orders_full.columns else 0
co2_series = orders_full[co2_col] if co2_col in orders_full.columns else 0
cost_series = orders_full[cost_col] if cost_col in orders_full.columns else 0

orders_full['Delivery_Efficiency'] = 1 - (distance_series * 0.4 + co2_series * 0.3 + cost_series * 0.3)
# normalize efficiency
try:
    orders_full['Delivery_Efficiency'] = MinMaxScaler().fit_transform(orders_full[['Delivery_Efficiency']])
except Exception:
    orders_full['Delivery_Efficiency'] = orders_full['Delivery_Efficiency']

# merge priority (if not present)
if 'Priority' not in orders_full.columns and 'Priority' in delivery_perf.columns:
    orders_full = orders_full.merge(delivery_perf[['Order_ID', 'Priority']], on='Order_ID', how='left')

# Priority score mapping
if 'Priority' in orders_full.columns:
    priority_map = {'Express': 3, 'Standard': 2, 'Economy': 1}
    orders_full['Priority_Score'] = orders_full['Priority'].map(priority_map)
else:
    orders_full['Priority_Score'] = 0

# numeric normalization for selected numeric cols (if exist)
numeric_candidates = []
for base in ['Distance_KM', 'Total_CO2_Kg', 'Delivery_Cost_INR', 'Priority_Score']:
    c = get_col(orders_full, base) or (base if base in orders_full.columns else None)
    if c:
        numeric_candidates.append(c)
if numeric_candidates:
    try:
        orders_full[numeric_candidates] = scaler.fit_transform(orders_full[numeric_candidates].fillna(0))
    except Exception:
        pass

# Assignment score (using safe references)
distance_for_score = orders_full[d_col] if d_col in orders_full.columns else 0
co2_for_score = orders_full[co2_col] if co2_col in orders_full.columns else 0
cost_for_score = orders_full[cost_col] if (cost_col and cost_col in orders_full.columns) else 0

weights = {'Priority_Score': 0.4, 'Distance_KM': 0.2, 'Total_CO2_Kg': 0.2, 'Delivery_Cost_INR': 0.2}
orders_full['Assignment_Score'] = (
    orders_full['Priority_Score'] * weights['Priority_Score'] -
    distance_for_score * weights['Distance_KM'] -
    co2_for_score * weights['Total_CO2_Kg'] -
    cost_for_score * weights['Delivery_Cost_INR']
)

orders_full = orders_full.sort_values(by='Assignment_Score', ascending=False)

# Final vehicle assignment (simple round-robin over available vehicles)
vehicles_available = available_vehicles.copy()
orders_full['Assigned_Vehicle'] = None
if 'Vehicle_ID' in vehicles_available.columns:
    vehicle_ids = vehicles_available['Vehicle_ID'].tolist()
else:
    vehicle_ids = vehicles_available.index.astype(str).tolist()

vidx = 0
for idx in orders_full.index:
    if not vehicle_ids:
        break
    orders_full.at[idx, 'Assigned_Vehicle'] = vehicle_ids[vidx % len(vehicle_ids)]
    vidx += 1

# merge vehicle CO2 for final calculation
if 'Vehicle_ID' in vehicles_df.columns and 'CO2_Emissions_Kg_per_KM' in vehicles_df.columns:
    orders_full = orders_full.merge(vehicles_df[['Vehicle_ID', 'CO2_Emissions_Kg_per_KM']], left_on='Assigned_Vehicle', right_on='Vehicle_ID', how='left')

# final Total_CO2_Kg using whichever distance column exists
final_distance_col = get_col(orders_full, "Distance_KM")
final_co2_em_col = get_col(orders_full, "CO2_Emissions_Kg_per_KM")
if final_distance_col and final_co2_em_col:
    orders_full['Total_CO2_Kg'] = orders_full[final_distance_col].fillna(0) * orders_full[final_co2_em_col].fillna(0)
else:
    # if distance exists but merged vehicle co2 column different, try fallback
    orders_full['Total_CO2_Kg'] = orders_full.get('Total_CO2_Kg', 0)

# ---------------- UI: Filters (now that orders_full is prepared) ----------------
# derive filter choices
priority_choices = list(orders_full['Priority'].dropna().unique()) if 'Priority' in orders_full.columns else []
vehicle_choices = list(orders_full['Assigned_Vehicle'].dropna().unique()) if 'Assigned_Vehicle' in orders_full.columns else []

selected_priority = st.sidebar.multiselect("Priority", options=priority_choices, default=priority_choices)
selected_vehicle = st.sidebar.multiselect("Assigned Vehicle", options=vehicle_choices, default=vehicle_choices)

# Date filter if exists
order_date_col = get_col(orders_full, "Order_Date")
if order_date_col:
    try:
        orders_full[order_date_col] = pd.to_datetime(orders_full[order_date_col])
        dmin = orders_full[order_date_col].min()
        dmax = orders_full[order_date_col].max()
        selected_date_range = st.sidebar.date_input("Order Date Range", [dmin.date(), dmax.date()])
    except Exception:
        order_date_col = None
        selected_date_range = None
else:
    selected_date_range = None

# apply filters to a view df
view_df = orders_full.copy()
if selected_priority:
    if 'Priority' in view_df.columns:
        view_df = view_df[view_df['Priority'].isin(selected_priority)]
if selected_vehicle:
    if 'Assigned_Vehicle' in view_df.columns:
        view_df = view_df[view_df['Assigned_Vehicle'].isin(selected_vehicle)]
if selected_date_range and order_date_col:
    view_df = view_df[
        (view_df[order_date_col].dt.date >= selected_date_range[0]) &
        (view_df[order_date_col].dt.date <= selected_date_range[1])
    ]

# ---------- OVERVIEW PAGE ----------
if page == "Overview":
    st.title("🚀 Smart Delivery Optimization Dashboard")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_orders = len(view_df)
    avg_eff = view_df['Delivery_Efficiency'].mean() if 'Delivery_Efficiency' in view_df.columns else 0
    avg_co2 = view_df['Total_CO2_Kg'].mean() if 'Total_CO2_Kg' in view_df.columns else 0
    delayed_orders = view_df['delayed'].sum() if 'delayed' in view_df.columns else 0

    kpi1.metric("📦 Total Orders", f"{total_orders}")
    kpi2.metric("🌿 Avg CO₂ Emission (Kg)", f"{avg_co2:.2f}")
    kpi3.metric("⚙️ Avg Efficiency", f"{avg_eff:.2f}")
    kpi4.metric("⏰ Delayed Deliveries", f"{delayed_orders}")

    st.markdown("---")

    if 'Delivery_Efficiency' in view_df.columns and 'Priority' in view_df.columns:
        fig_eff = px.box(
            view_df, x="Priority", y="Delivery_Efficiency",
            title="📊 Delivery Efficiency by Priority",
            color="Priority", color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_eff.update_layout(template="plotly_dark")
        st.plotly_chart(fig_eff, use_container_width=True)

    if 'Distance_KM' in view_df.columns and 'Total_CO2_Kg' in view_df.columns:
        fig_scatter = px.scatter(
            view_df, x='Distance_KM', y='Total_CO2_Kg',
            color='Priority' if 'Priority' in view_df.columns else None,
            title="🌍 Distance vs CO₂ Emissions",
            size='Delivery_Efficiency' if 'Delivery_Efficiency' in view_df.columns else None,
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_scatter.update_layout(template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- VISUALIZATIONS PAGE ----------
elif page == "Visualizations":
    st.title("📈 Advanced Visualizations")

    if 'Assignment_Score' in view_df.columns:
        fig_hist = px.histogram(
            view_df, x='Assignment_Score', nbins=30,
            title="📦 Distribution of Assignment Scores",
            color_discrete_sequence=["#29b6f6"]
        )
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

    if 'Assigned_Vehicle' in view_df.columns:
        vehicle_perf = view_df.groupby('Assigned_Vehicle').agg({
            'Delivery_Efficiency': 'mean',
            'Total_CO2_Kg': 'mean'
        }).reset_index()
        fig_bar = px.bar(
            vehicle_perf, x='Assigned_Vehicle', y='Delivery_Efficiency',
            color='Total_CO2_Kg',
            title="🚛 Vehicle Efficiency vs CO₂ Output",
            color_continuous_scale="Viridis"
        )
        fig_bar.update_layout(template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------- INSIGHTS PAGE ----------
elif page == "Insights":
    st.title("🧠 Insights & Analysis")

    if report:
        st.subheader("Model Performance Report")
        st.text(report)
    else:
        st.info("No model performance report available.")

    st.markdown("#### Top 10 Most Efficient Deliveries")
    if 'Delivery_Efficiency' in view_df.columns:
        st.dataframe(view_df.nlargest(10, 'Delivery_Efficiency')[
            ['Order_ID', 'Delivery_Efficiency', 'Assigned_Vehicle', 'Priority']
        ])

    st.markdown("#### Least Efficient Deliveries")
    if 'Delivery_Efficiency' in view_df.columns:
        st.dataframe(view_df.nsmallest(10, 'Delivery_Efficiency')[
            ['Order_ID', 'Delivery_Efficiency', 'Assigned_Vehicle', 'Priority']
        ])

# ---------- EXPORT PAGE ----------
elif page == "Export":
    st.title("📤 Export Data")
    st.write("You can download the processed and scored dataset below.")
    csv = view_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="optimized_deliveries.csv", mime="text/csv")
    st.success("✅ Export ready!")

# ---------- Footer ----------
st.markdown("---")
st.caption("✨ Developed with ❤️ by Akshara | Powered by Streamlit & Plotly ✨")

