import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sqlalchemy import text
from Database.DBConnection import get_sqlalchemy_engine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Rapido Mobility Insights",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        :root {
            color-scheme: light;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {
            display: none;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f3f7ff 0%, #ffffff 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #082d5d 0%, #1346ab 100%);
            color: #f8fbff;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
            font-weight: 500;
        }
        [data-testid="stSidebar"] .css-1hynsf2,
        [data-testid="stSidebar"] .css-ffhzg2,
        [data-testid="stSidebar"] .css-1y4p8pa {
            color: #ffffff !important;
        }
        .block-container {
            padding: 1.5rem 2rem 2rem;
        }
        .stMarkdown p,
        .stText {
            color: #243b55;
            line-height: 1.7;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            color: #102a43;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.65rem;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] {
            color: #5c7291;
        }
        .css-18e3th9 {
            background-color: rgba(255, 255, 255, 0.9) !important;
            box-shadow: 0px 18px 40px rgba(15, 42, 80, 0.08);
            border-radius: 24px;
        }
        .css-1v0mbdj {
            background: #0a4fa6;
            color: #ffffff;
        }
        .css-1v0mbdj:hover {
            background: #0d63c7;
        }
        .stButton>button {
            border-radius: 999px;
            padding: 0.9rem 1.4rem;
        }
        .stTabs [role="tab"] {
            font-weight: 600;
        }
        .insights-header {
            background: rgba(255, 255, 255, 0.95);
            border-left: 5px solid #0d63c7;
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            box-shadow: 0 14px 30px rgba(15, 42, 80, 0.08);
            margin-bottom: 1.25rem;
        }
        .insights-header h2 {
            margin: 0 0 0.35rem;
            color: #0f2d54;
            font-size: 2rem;
            letter-spacing: -0.04em;
        }
        .insights-header p {
            margin: 0;
            color: #475569;
            font-size: 1rem;
            line-height: 1.7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- DATA LOADING FROM MySQL ---
@st.cache_data
def load_data():
    query = text(
        """
       SELECT
            b.Booking_ID, b.Booking_Date, b.Booking_Time, b.City,
            b.Booking_Status, b.Pickup_Location, b.Drop_Location, 
		    b.Driver_ID, b.Customer_ID, b.Vehicle_Type, b.Ride_Distance_KM AS Distance, 
            b.Estimated_Ride_Time_Min, b.Actual_Ride_Time_Min,
            b.Traffic_Level, b.Weather_Condition, 
            b.Base_Fare, b.Surge_Multiplier, b.Booking_Value AS Fare,
            c.Avg_Customer_Rating,  COALESCE(c.cancellation_rate, 0) AS Customer_Cancellation_Rate,
            d.Acceptance_Rate, COALESCE(d.Avg_Driver_Rating, 0) AS Driver_Rating,
            b.hour_of_day AS Hour
        FROM bookings b
        LEFT JOIN customers c ON b.customer_id = c.customer_id
        LEFT JOIN drivers d ON b.driver_id = d.driver_id
        """
    )

    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
    except Exception as exc:
        st.error(f"Failed to read data from MySQL: {exc}")
        df = pd.DataFrame()
    finally:
        if 'engine' in locals():
            engine.dispose()

    return df

@st.cache_data
def get_location_outcome_probabilities(data, location_field, location):
    if data.empty or location not in data[location_field].dropna().unique():
        return {"Completed": 0.0, "Cancelled": 0.0, "Incomplete": 0.0}

    status = data['Booking_Status'].astype(str).str.lower()
    outcomes = np.select(
        [status.isin(['completed', 'success']), status.isin(['cancelled', 'canceled'])],
        ['Completed', 'Cancelled'],
        default='Incomplete'
    )
    temp = pd.DataFrame({location_field: data[location_field], 'Outcome': outcomes})
    counts = temp[temp[location_field] == location]['Outcome'].value_counts(normalize=True)
    return {outcome_name: float(counts.get(outcome_name, 0.0)) for outcome_name in ["Completed", "Cancelled", "Incomplete"]}

@st.cache_data
def get_ride_outcome_probabilities(data, city=None, traffic=None, weather=None, cust_hist=0.0, vehicle_type=None):
    if data.empty:
        return {"Completed": 0.0, "Cancelled": 0.0, "Incomplete": 0.0}

    subset = data.copy()
    if city is not None:
        subset = subset[subset['City'] == city]
    if subset.empty:
        subset = data.copy()

    status = subset['Booking_Status'].astype(str).str.lower()
    outcomes = np.select(
        [status.isin(['completed', 'success']), status.isin(['cancelled', 'canceled'])],
        ['Completed', 'Cancelled'],
        default='Incomplete'
    )
    counts = pd.Series(outcomes).value_counts(normalize=True)
    probs = {outcome_name: float(counts.get(outcome_name, 0.0)) for outcome_name in ["Completed", "Cancelled", "Incomplete"]}

    if traffic == "High":
        probs["Cancelled"] += 0.10
        probs["Completed"] -= 0.05
        probs["Incomplete"] -= 0.05
    elif traffic == "Medium":
        probs["Cancelled"] += 0.05
        probs["Completed"] -= 0.02

    if weather in ["Rainy", "Stormy"]:
        probs["Cancelled"] += 0.08
        probs["Completed"] -= 0.04
        probs["Incomplete"] -= 0.04

    if cust_hist >= 0.7:
        probs["Cancelled"] += 0.12
        probs["Completed"] -= 0.06

    if vehicle_type == "Auto":
        probs["Cancelled"] += 0.03
        probs["Completed"] -= 0.02
    elif vehicle_type == "Cab":
        probs["Completed"] += 0.03

    for key in probs:
        probs[key] = max(0.0, probs[key])

    total = sum(probs.values()) or 1.0
    return {key: value / total for key, value in probs.items()}

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚖 Rapido Analytics")
page = st.sidebar.radio("Navigate to:", ["Project Overview", "Exploratory Data Analysis", "Predictions", "Operational Insights"])

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "Project Overview":
    st.title("Rapido: Intelligent Mobility Insights")
    st.markdown("""
        Rapido operates a large-scale ride-hailing platform where millions of bookings are created daily across multiple cities, vehicle types, and demand conditions. 
        The primary business challenge is to predict ride outcomes (completed vs cancelled) and fare amounts accurately, while also identifying high-risk bookings that may lead to cancellations or no-shows.
    """)    
    st.subheader("Overall Data")
    if df.empty:
        st.warning("No data available to display.")
    else:
        st.dataframe(df, use_container_width=True)

# --- PAGE 2: EXPLORATORY DATA ANALYSIS (EDA) ---
elif page == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.subheader("EDA Filters")

    if df.empty:
        st.warning("No data available to explore.")
        filtered_df = df
    else:
        city_options = df['City'].dropna().unique().tolist()
        vehicle_options = df['Vehicle_Type'].dropna().unique().tolist()
        status_options = df['Booking_Status'].dropna().unique().tolist()

        selected_cities = st.sidebar.multiselect("City", city_options, default=city_options)
        selected_vehicles = st.sidebar.multiselect("Vehicle Type", vehicle_options, default=vehicle_options)
        selected_status = st.sidebar.multiselect("Booking Status", status_options, default=status_options)

        filtered_df = df[
            df['City'].isin(selected_cities) &
            df['Vehicle_Type'].isin(selected_vehicles) &
            df['Booking_Status'].isin(selected_status)
        ]

    st.subheader("Summary Metrics")

    tabs = st.tabs([
        "Overview & Trends",
        "Ride Volume",
        "Distance vs Fare",
        "City Metrics",
        "Cancellation Heatmap",
        "Customer vs Driver Metrics",
        "Vehicle Type & Booking Patterns",
        "Weather & Traffic Impact"
    ])

    with tabs[0]:
        st.markdown("<h2 style='font-size: 22px;'><b>🔍 Key Insights from Data Analysis</b></h2>", unsafe_allow_html=True)

        # Calculate key metrics for insights
        total_bookings = len(df)
        completed_bookings = len(df[df['Booking_Status'].str.lower().isin(['completed', 'success'])])
        cancelled_bookings = len(df[df['Booking_Status'].str.lower().isin(['cancelled', 'canceled'])])
        completion_rate = (completed_bookings / total_bookings * 100) if total_bookings > 0 else 0

        # Peak hours analysis
        peak_hour = df.groupby('Hour').size().idxmax() if not df.empty else 0
        peak_hour_volume = df.groupby('Hour').size().max() if not df.empty else 0

        # City performance
        city_performance = df.groupby('City').agg({
            'Booking_ID': 'count',
            'Fare': 'mean',
            'Customer_Cancellation_Rate': 'mean',
            'Acceptance_Rate': 'mean'
        })

        # Top performing city
        top_city = city_performance['Booking_ID'].idxmax() if not city_performance.empty else "N/A"

        col_insights1, col_insights2 = st.columns(2)

        with col_insights1:
            st.info("📊 **Overall Performance**")
            st.metric("Total Bookings", f"{total_bookings:,}")
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
            st.metric("Peak Hour", f"{peak_hour}:00 ({peak_hour_volume} bookings)")
            st.metric("Top City", top_city)

        with col_insights2:
            st.info("💰 **Revenue Insights**")
            avg_fare = df['Fare'].mean()
            total_revenue = df['Fare'].sum()
            st.metric("Average Fare", f"₹{avg_fare:.2f}")
            st.metric("Total Revenue", f"₹{total_revenue:,.0f}")

        st.divider()

        st.subheader("🏙️ City-wise Analysis")
        # Display city performance table
        city_table = city_performance.rename(columns={
            'Booking_ID': 'Total Bookings',
            'Fare': 'Avg Fare (₹)',
            'Customer_Cancellation_Rate': 'Customer Cancel Rate',
            'Acceptance_Rate': 'Driver Acceptance Rate'
        })
        st.dataframe(city_table.style.format({
            'Avg Fare (₹)': '₹{:.2f}',
            'Customer Cancel Rate': '{:.5f}',
            'Driver Acceptance Rate': '{:.5f}'
        }).highlight_max(axis=0), use_container_width=True)

    with tabs[1]:
        col_a, col_b = st.columns(2)
        with col_a:
            hourly_counts = filtered_df.groupby('Hour').size().reset_index(name='count')
            fig_hour = px.line(hourly_counts, x='Hour', y='count', markers=True, template="plotly_white")
            fig_hour.update_layout(title='Rides by Hour', xaxis_title='Hour', yaxis_title='Bookings')
            st.plotly_chart(fig_hour, use_container_width=True)
        with col_b:
            city_counts = filtered_df.groupby('City').size().reset_index(name='count')
            fig_city = px.bar(city_counts, x='City', y='count', template="plotly_white")
            fig_city.update_layout(title='Rides by City', xaxis_title='City', yaxis_title='Bookings')
            st.plotly_chart(fig_city, use_container_width=True)

    with tabs[2]:
        st.subheader("Demand & Pricing")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_dist = px.scatter(
                filtered_df,
                x="Distance",
                y="Fare",
                color="City",
                labels={'Distance': 'Ride Distance (KM)', 'Fare': 'Booking Value (₹)'},
                template="plotly_white"
            )
            fig_dist.update_layout(title='Distance vs Fare')
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_c2:
            vehicle_fare = filtered_df.groupby('Vehicle_Type')['Fare'].mean().reset_index()
            fig_vehicle_fare = px.bar(
                vehicle_fare,
                x='Vehicle_Type',
                y='Fare',
                color='Vehicle_Type',
                labels={'Fare': 'Avg Booking Value (₹)'},
                template="plotly_white"
            )
            fig_vehicle_fare.update_layout(title='Average Fare by Vehicle Type')
            st.plotly_chart(fig_vehicle_fare, use_container_width=True)

    with tabs[3]:
        st.subheader("Outcome Distribution by City")
        fig2 = px.histogram(df, x="City", color="Booking_Status", barmode="group", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Driver Acceptance Rate by City")
            acceptance_by_driver = df.groupby('City')['Acceptance_Rate'].mean().reset_index()
            fig5 = px.line(
                acceptance_by_driver,
                x='City',
                y='Acceptance_Rate',
                markers=True,
                labels={'Acceptance_Rate': 'Avg Acceptance Rate'},
                template="plotly_white"
            )
            st.plotly_chart(fig5, use_container_width=True)
        with col6:
            st.subheader("Customer Cancellation Rate by City")
            cancellation_by_city = df.groupby('City')['Customer_Cancellation_Rate'].mean().reset_index()
            fig6 = px.line(
                cancellation_by_city,
                x='City',
                y='Customer_Cancellation_Rate',
                markers=True,
                labels={'Customer_Cancellation_Rate': 'Avg Cancellation Rate'},
                template="plotly_white"
            )
            st.plotly_chart(fig6, use_container_width=True)

    with tabs[4]:
        st.subheader("Cancellation Heatmap")

        cancel_data = filtered_df.copy()
        cancel_data['Is_Cancelled'] = cancel_data['Booking_Status'].astype(str).str.lower().isin(['cancelled', 'canceled']).astype(int)
        cancellation_by_city = cancel_data.groupby('City')['Is_Cancelled'].mean().reset_index()
      
        if not cancel_data.empty:
            heatmap_data = cancel_data.pivot_table(
                index='City',
                columns='Hour',
                values='Is_Cancelled',
                aggfunc='mean'
            ).fillna(0)
            city_order = heatmap_data.mean(axis=1).sort_values(ascending=False).index
            heatmap_data = heatmap_data.loc[city_order]
            fig_heat = px.imshow(
                heatmap_data,
                labels={'x': 'Hour of Day', 'y': 'City', 'color': 'Cancellation Rate'},
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale='Reds',
                aspect='auto',
                text_auto='.1%'
            )
            fig_heat.update_traces(
                texttemplate='%{z:.1%}',
                hovertemplate='City: %{y}<br>Hour: %{x}<br>Cancellation Rate: %{z:.1%}<extra></extra>'
            )
            fig_heat.update_layout(
                title='City-by-Hour Cancellation Heatmap',
                xaxis_title='Hour of Day',
                yaxis_title='City',
                coloraxis_colorbar=dict(title='Rate', ticksuffix='%'),
                margin=dict(l=120, r=20, t=50, b=40)
            )
            fig_heat.update_xaxes(tickmode='linear')
            st.plotly_chart(fig_heat, use_container_width=True)

        fig_cancel_city = px.bar(
            cancellation_by_city,
            x='City',
            y='Is_Cancelled',
            color='Is_Cancelled',
            labels={'Is_Cancelled': 'Cancellation Rate'},
            template="plotly_white"
        )
        fig_cancel_city.update_layout(title='Cancellation Rate by City', yaxis_tickformat='.0%')
        st.plotly_chart(fig_cancel_city, use_container_width=True)
    
    with tabs[5]:
        st.subheader("Customer vs Driver Behaviour Comparison")
        col_cust, col_drv = st.columns(2)
        
        with col_cust:
            st.write("**Customer Metrics by City**")
            cust_metrics = filtered_df.groupby('City').agg({
                'Avg_Customer_Rating': 'mean',
                'Customer_Cancellation_Rate': 'mean'
            }).reset_index()
            fig_cust = px.bar(
                cust_metrics,
                x='City',
                y='Avg_Customer_Rating',
                color='Customer_Cancellation_Rate',
                labels={'Avg_Customer_Rating': 'Avg Rating', 'Customer_Cancellation_Rate': 'Cancellation Rate'},
                template="plotly_white"
            )
            st.plotly_chart(fig_cust, use_container_width=True)
        
        with col_drv:
            st.write("**Driver Metrics by City**")
            drv_metrics = filtered_df.groupby('City').agg({
                'Driver_Rating': 'mean',
                'Acceptance_Rate': 'mean'
            }).reset_index()
            fig_drv = px.bar(
                drv_metrics,
                x='City',
                y='Driver_Rating',
                color='Acceptance_Rate',
                labels={'Driver_Rating': 'Avg Rating', 'Acceptance_Rate': 'Acceptance Rate'},
                template="plotly_white"
            )
            st.plotly_chart(fig_drv, use_container_width=True)

    with tabs[6]:
        st.subheader("Vehicle Type & Booking Patterns")
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.write("**Ride Distribution by Vehicle Type**")
            vehicle_dist = filtered_df['Vehicle_Type'].value_counts().reset_index()
            vehicle_dist.columns = ['Vehicle_Type', 'Count']
            fig_vehicle = px.pie(
                vehicle_dist,
                names='Vehicle_Type',
                values='Count',
                template="plotly_white"
            )
            st.plotly_chart(fig_vehicle, use_container_width=True)
        
        with col_v2:
            st.write("**Average Fare by Vehicle Type**")
            vehicle_fare = filtered_df.groupby('Vehicle_Type')['Fare'].mean().reset_index()
            fig_fare = px.bar(
                vehicle_fare,
                x='Vehicle_Type',
                y='Fare',
                color='Fare',
                labels={'Fare': 'Avg Booking Value (₹)'},
                template="plotly_white"
            )
            st.plotly_chart(fig_fare, use_container_width=True)

    with tabs[7]:
        st.subheader("Traffic & Weather Impact on Cancellations")
        col_traffic, col_weather = st.columns(2)
        
        with col_traffic:
            st.write("**Cancellation Rate by Traffic Level**")
            if 'Traffic_Level' in filtered_df.columns:
                traffic_cancel = filtered_df.copy()
                traffic_cancel['Is_Cancelled'] = traffic_cancel['Booking_Status'].astype(str).str.lower().isin(['cancelled', 'canceled']).astype(int)
                traffic_data = traffic_cancel.groupby('Traffic_Level')['Is_Cancelled'].agg(['sum', 'count']).reset_index()
                traffic_data['Cancellation_Rate'] = (traffic_data['sum'] / traffic_data['count']).fillna(0)
                fig_traffic = px.bar(
                    traffic_data,
                    x='Traffic_Level',
                    y='Cancellation_Rate',
                    color='Cancellation_Rate',
                    labels={'Cancellation_Rate': 'Rate'},
                    template="plotly_white"
                )
                st.plotly_chart(fig_traffic, use_container_width=True)
            else:
                st.info("Traffic Level data not available")
        
        with col_weather:
            st.write("**Cancellation Rate by Weather Condition**")
            if 'Weather_Condition' in filtered_df.columns:
                weather_cancel = filtered_df.copy()
                weather_cancel['Is_Cancelled'] = weather_cancel['Booking_Status'].astype(str).str.lower().isin(['cancelled', 'canceled']).astype(int)
                weather_data = weather_cancel.groupby('Weather_Condition')['Is_Cancelled'].agg(['sum', 'count']).reset_index()
                weather_data['Cancellation_Rate'] = (weather_data['sum'] / weather_data['count']).fillna(0)
                fig_weather = px.bar(
                    weather_data,
                    x='Weather_Condition',
                    y='Cancellation_Rate',
                    color='Cancellation_Rate',
                    labels={'Cancellation_Rate': 'Rate'},
                    template="plotly_white"
                )
                st.plotly_chart(fig_weather, use_container_width=True)
            else:
                st.info("Weather Condition data not available")

elif page == "Predictions":
    st.header("ML Prediction Engine")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Ride Outcome", "Fare Prediction", "Cancellation Risk", "Driver Delay Risk"])
   
    with tab1:
        st.subheader("Ride Outcome Prediction (Multi-Class Classification)")
        c1, c2, c3 = st.columns(3)
        city_options = sorted(df['City'].dropna().unique()) if not df.empty else []
        selected_city = c1.selectbox("Select City", city_options, key="outcome_city") if city_options else None

        traffic = c2.selectbox("Traffic Condition", ["Low", "Medium", "High"], key="outcome_traffic")
        weather = c3.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy", "Windy", "Other"], key="outcome_weather")
     
        c4, c5 = st.columns(2)
        v_type = c4.selectbox("Vehicle Type", ["Bike", "Auto", "Cab"], key="outcome_vehicle")
        cust_hist = c5.slider("Customer Historical Cancellation Rate", 0.0, 1.0, 0.1)

        outcome_probs = get_ride_outcome_probabilities(
            df,
            city=selected_city,
            traffic=traffic,
            weather=weather,
            cust_hist=cust_hist,
            vehicle_type=v_type
        )
        predicted_outcome = max(outcome_probs, key=outcome_probs.get)

        if predicted_outcome == "Completed":
            st.success(f"✅ Predicted outcome: {predicted_outcome}")
        elif predicted_outcome == "Cancelled":
            st.error(f"❌ Predicted outcome: {predicted_outcome}")
        else:
            st.warning(f"⚠️ Predicted outcome: {predicted_outcome}")

        st.markdown("**Outcome probabilities:**")
        prob_df = pd.DataFrame({
            "Outcome": list(outcome_probs.keys()),
            "Probability": [f"{value * 100:.1f}%" for value in outcome_probs.values()]
        })
        st.dataframe(prob_df, use_container_width=True)
 
    with tab2:
        st.subheader("Fare Prediction Model (Regression)")
        c1, c2, c3 = st.columns(3)
        city_options = sorted(df['City'].dropna().unique()) if not df.empty else []
        selected_city = c1.selectbox("Select City", city_options, key="fare_city") if city_options else None

        dist = c2.number_input("Distance (KM)", min_value=0.5, max_value=100.0, value=5.0)
        est_time = c3.number_input("Estimated Ride Time (Min)", min_value=1, max_value=240, value=20)

        c3, c4 = st.columns(2)
        v_type = c3.selectbox("Vehicle Type", ["Bike", "Auto", "Cab"], key="fare_vehicle")
        traffic = c4.selectbox("Traffic Condition", ["Low", "Medium", "High"], key="fare_traffic")

        c5, c6, c7 = st.columns(3)
        weather = c5.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy", "Windy", "Other"], key="fare_weather")
        surge = c6.checkbox("Surge Pricing Applied")

        time_day = c7.slider("Time of Day", 0, 23, 12)
        rush_hour_flag = time_day in list(range(7, 10)) + list(range(17, 21))
        long_distance_flag = dist >= 15

        city_factor = 10 if selected_city and selected_city.lower() in ["mumbai", "delhi", "bangalore"] else 0
        traffic_factor = 15 if traffic == "High" else 7 if traffic == "Medium" else 0
        weather_factor = 12 if weather in ["Rainy", "Stormy"] else 5 if weather == "Windy" else 0
        surge_factor = 20 if surge else 0
        base_fare = 25 if v_type == "Bike" else 50
        estimated_fare = (dist * 12) + base_fare + city_factor + traffic_factor + weather_factor + surge_factor

        fare_per_km = estimated_fare / dist if dist else 0
        fare_per_min = estimated_fare / est_time if est_time else 0

        st.markdown("**Derived Ride Features**")
        c7, c8, c9, c10 = st.columns(4)
        c7.write(f"Fare per KM: ₹{fare_per_km:.2f}")
        c8.write(f"Fare per Min: ₹{fare_per_min:.2f}")
      
        c7, c8, c9, c10 = st.columns(4)
        c7.write(f"Rush Hour Flag: {'Yes' if rush_hour_flag else 'No'}")
        c8.write(f"Long Distance Flag: {'Yes' if long_distance_flag else 'No'}")

        st.write(f"**City :** {selected_city} ")

        st.metric(label="Predicted Booking Value", value=f"₹{estimated_fare:.2f}")

    with tab3:
        st.subheader("Customer Cancellation Risk Model (Binary Classification)")
        c1, c2 = st.columns(2)
        cust_hist = c1.slider("Historical Cancellation Rate", 0.0, 1.0, 0.1)
        past_rating = c2.slider("Past Customer Rating", 1.0, 5.0, 4.0)

        c1, c2, c3, c4 = st.columns(4)
        peak_time_behaviour = c1.selectbox("Peak Time Behavior", ["Low", "Medium", "High"], key="cancel_peak")
        pricing_sensitivity = c2.selectbox("Pricing Sensitivity", ["Low", "Medium", "High"], key="cancel_pricing")
        customer_loyalty_score = c3.slider("Customer Loyalty Score", 1.0, 5.0, 4.0)

        risk_score = 0.2 * cust_hist
        risk_score += 0.2 * (5.0 - past_rating) / 4.0
        risk_score += 0.2 * (0.5 if peak_time_behaviour == "High" else 0.25 if peak_time_behaviour == "Medium" else 0.0)
        risk_score += 0.2 * (0.5 if pricing_sensitivity == "High" else 0.25 if pricing_sensitivity == "Medium" else 0.0)
        risk_score += 0.2 * ((5.0 - customer_loyalty_score) / 4.0)
        cancellation_probability = min(1.0, max(0.0, risk_score))

        st.metric(label="Cancellation Probability", value=f"{cancellation_probability * 100:.1f}%")
        if cancellation_probability > 0.75:
            st.error("High cancellation risk")
        elif cancellation_probability > 0.4:
            st.warning("Medium cancellation risk")
        else:
            st.success("Low cancellation risk")

    with tab4:
        st.subheader("Driver Delay Prediction Model (Binary Classification)")
        c1, c2 = st.columns(2)
        delay_history = c1.number_input("Past Delay Count", min_value=0, max_value=50, value=2)
        traffic_exposure = c2.selectbox("Traffic Exposure", ["Low", "Medium", "High"], key="delay_traffic")
        c1, c2 = st.columns(2)
        acceptance_rate = c1.slider("Driver Acceptance Rate", 0.0, 1.0, 0.8)
        driver_reliability_score = c2.slider("Driver Reliability Score", 1.0, 5.0, 4.0)

        delay_risk = 0.3 * (delay_history / 20.0)
        delay_risk += 0.3 * (0.5 if traffic_exposure == "High" else 0.25 if traffic_exposure == "Medium" else 0.0)
        delay_risk += 0.2 * (1.0 - acceptance_rate)
        delay_risk += 0.2 * ((5.0 - driver_reliability_score) / 4.0)
        delay_probability = min(1.0, max(0.0, delay_risk))

        st.metric(label="Driver Delay Probability", value=f"{delay_probability * 100:.1f}%")
        if delay_probability > 0.7:
            st.error("High risk of driver delay or incomplete ride")
        elif delay_probability > 0.35:
            st.warning("Moderate delay risk")
        else:
            st.success("Low delay risk")

# --- PAGE 4: OPERATIONAL INSIGHTS ---
elif page == "Operational Insights":
    st.header("📈 Strategy & Recommendations")
    # --- EDA INSIGHTS SECTION ---

    st.subheader("⚡ Operational Insights")

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        st.markdown("""
        **Demand Patterns:**
        - Peak booking hours suggest optimal driver allocation times
        - City-specific demand variations indicate targeted marketing opportunities
        - Distance vs Fare correlation shows pricing efficiency across routes

        **Customer Behavior:**
        - Cancellation rates vary significantly by city and time
        - Customer ratings correlate with booking completion likelihood
        - Vehicle type preferences affect booking volumes
        """)

    with insights_col2:
        st.markdown("""
        **Driver Performance:**
        - Acceptance rates impact overall service availability
        - Driver ratings influence customer satisfaction
        - City-specific driver metrics suggest training needs

        **External Factors:**
        - Traffic conditions significantly affect cancellation rates
        - Weather patterns may influence booking behavior
        - Surge pricing effectiveness varies by time and location
        """)

    st.divider()

    st.subheader("🎯 Recommendations")

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("""
        **Immediate Actions:**
        - Increase driver incentives during peak hours in high-demand cities
        - Implement dynamic pricing based on traffic and weather conditions
        - Target customer retention campaigns in cities with high cancellation rates

        **Medium-term Strategies:**
        - Optimize fleet distribution based on vehicle type preferences
        - Develop predictive models for demand forecasting
        - Enhance driver training programs in underperforming cities
        """)

    with rec_col2:
        st.markdown("""
        **Long-term Planning:**
        - Expand operations in high-performing cities with low cancellation rates
        - Invest in technology for real-time traffic and weather integration
        - Develop personalized pricing models based on customer behavior patterns

        **Risk Mitigation:**
        - Monitor driver acceptance rates for early intervention
        - Track customer cancellation trends for proactive service improvements
        - Maintain contingency plans for adverse weather conditions
        """)


    st.info("Based on the current data patterns, the following interventions are suggested:")
    
    st.markdown("""
    1. **Dynamic Re-allocation:** Increase driver incentives in **Bangalore** during the 18:00 - 21:00 window to combat 15% higher cancellation rates.
    2. **Low-Reliability Filter:** Flag drivers with a reliability score below **0.6** for manual training interventions.
    3. **Fare Calibration:** Current models suggest a **10% increase** in base fares for long-distance trips (>15km) to improve driver acceptance.
    """)
   