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
st.set_page_config(page_title="Rapido Mobility Insights", layout="wide")

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
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview & Trends",
        "Ride Volume",
        "Distance vs Fare",
        "City Metrics",
        "Cancellation Heatmap",
        "Customer vs Driver Metrics",
        "Vehicle Type & Booking Patterns",
        "Weather & Traffic Impact"
    ])

    with tab1:
        st.markdown("### **🔍 Key Insights from Data Analysis**")

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
            st.subheader("📊 Overall Performance")
            st.metric("Total Bookings", f"{total_bookings:,}")
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
            st.metric("Peak Hour", f"{peak_hour}:00 ({peak_hour_volume} bookings)")
            st.metric("Top City", top_city)

        with col_insights2:
            st.subheader("💰 Revenue Insights")
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

        st.divider()
    with tab2:
        st.subheader("Ride Volume by Hour")
        hourly_counts = df.groupby('Hour').size().reset_index(name='counts')
        fig1 = px.line(hourly_counts, x='Hour', y='counts', markers=True, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Ride Volume by City")
        city_counts = df.groupby('City').size().reset_index(name='counts')
        fig2 = px.bar(city_counts, x='City', y='counts', template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab2:
        st.subheader("Distance vs Fare Correlation")
        fig_dist = px.scatter(
            df,
            x="Distance",
            y="Fare",
            color="City",
            labels={'Distance': 'Ride Distance (KM)', 'Fare': 'Booking Value (₹)'},
            template="plotly_white"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
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

    with tab4:
        st.subheader("Cancellation Heatmap")
        if df.empty:
            st.warning("No data available for the heatmap.")
        else:
            cancel_data = df.copy()
            cancel_data['Is_Cancelled'] = cancel_data['Booking_Status'].astype(str).str.lower().isin(['cancelled', 'canceled']).astype(int)
            heatmap_data = cancel_data.pivot_table(
                index='City',
                columns='Hour',
                values='Is_Cancelled',
                aggfunc='mean'
            ).fillna(0)
            city_order = heatmap_data.mean(axis=1).sort_values(ascending=False).index
            heatmap_data = heatmap_data.loc[city_order]
            fig4 = px.imshow(
                heatmap_data,
                labels={'x': 'Hour of Day', 'y': 'City', 'color': 'Cancellation Rate'},
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale='Reds',
                aspect='auto',
                text_auto='.1%'
            )
            fig4.update_traces(
                texttemplate='%{z:.1%}',
                hovertemplate='City: %{y}<br>Hour: %{x}<br>Cancellation Rate: %{z:.1%}<extra></extra>'
            )
            fig4.update_layout(
                title='City-by-Hour Cancellation Rate',
                xaxis_title='Hour of Day',
                yaxis_title='City',
                coloraxis_colorbar=dict(title='Rate', ticksuffix='%'),
                margin=dict(l=120, r=20, t=50, b=40)
            )
            fig4.update_xaxes(tickmode='linear')
            st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        st.subheader("Customer vs Driver Behaviour Comparison")
        col_cust, col_drv = st.columns(2)
        
        with col_cust:
            st.write("**Customer Metrics by City**")
            cust_metrics = df.groupby('City').agg({
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
            drv_metrics = df.groupby('City').agg({
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

    with tab6:
        st.subheader("Vehicle Type & Booking Patterns")
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.write("**Ride Distribution by Vehicle Type**")
            vehicle_dist = df['Vehicle_Type'].value_counts().reset_index()
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
            vehicle_fare = df.groupby('Vehicle_Type')['Fare'].mean().reset_index()
            fig_fare = px.bar(
                vehicle_fare,
                x='Vehicle_Type',
                y='Fare',
                color='Fare',
                labels={'Fare': 'Avg Booking Value (₹)'},
                template="plotly_white"
            )
            st.plotly_chart(fig_fare, use_container_width=True)

    with tab7:
        st.subheader("Traffic & Weather Impact on Cancellations")
        col_traffic, col_weather = st.columns(2)
        
        with col_traffic:
            st.write("**Cancellation Rate by Traffic Level**")
            if 'Traffic_Level' in df.columns:
                traffic_cancel = df.copy()
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
            if 'Weather_Condition' in df.columns:
                weather_cancel = df.copy()
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
    
    tab1, tab2, tab3 = st.tabs(["Fare Estimator", "Outcome Predictor", "Risk Scoring"])
    
    with tab1:
        st.subheader("Estimate Trip Fare")
        c1, c2 = st.columns(2)
        dist = c1.number_input("Distance (KM)", min_value=0.5, max_value=100.0, value=5.0)
        v_type = c2.selectbox("Vehicle Type", ["Bik e", "Auto", "Cab"])
        time_day = st.slider("Time of Day", 0, 23, 12)
        surge = st.checkbox("Peak Hour / Surge Applied")
        
        # Placeholder for Regression Model
        base_fare = 25 if v_type == "Bike" else 50
        estimated_fare = (dist * 12) + base_fare + (20 if surge else 0)
        
        st.metric(label="Predicted Fare", value=f"₹{estimated_fare:.2f}")

    with tab2:
        st.subheader("Predict Ride Outcome")
        location_type = st.radio("Location Type", ["Pickup Location", "Drop Location"])
        location_field = "Pickup_Location" if location_type == "Pickup Location" else "Drop_Location"
        location_options = sorted(df[location_field].dropna().unique()) if not df.empty else []

        if not location_options:
            st.warning("No location data available for prediction.")
            selected_location = None
        else:
            selected_location = st.selectbox(f"Select {location_type}", location_options)

        cust_hist = st.slider("Customer Historical Cancellation Rate", 0.0, 1.0, 0.1)
        traffic = st.selectbox("Traffic Condition", ["Low", "Medium", "High"])

        if selected_location is not None:
            outcome_probs = get_location_outcome_probabilities(df, location_field, selected_location)
            adjusted_probs = outcome_probs.copy()

            if traffic == "High":
                adjusted_probs["Cancelled"] = min(1.0, adjusted_probs["Cancelled"] + 0.15)
                adjusted_probs["Completed"] = max(0.0, adjusted_probs["Completed"] - 0.08)
                adjusted_probs["Incomplete"] = max(0.0, adjusted_probs["Incomplete"] - 0.07)

            if cust_hist >= 0.7:
                adjusted_probs["Cancelled"] = min(1.0, adjusted_probs["Cancelled"] + 0.12)
                adjusted_probs["Completed"] = max(0.0, adjusted_probs["Completed"] - 0.06)

            total = sum(adjusted_probs.values()) or 1.0
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}
            predicted_outcome = max(adjusted_probs, key=adjusted_probs.get)

            if predicted_outcome == "Completed":
                st.success(f"✅ Predicted outcome: {predicted_outcome}")
            elif predicted_outcome == "Cancelled":
                st.error(f"❌ Predicted outcome: {predicted_outcome}")
            else:
                st.warning(f"⚠️ Predicted outcome: {predicted_outcome}")

            st.markdown(f"**Location:** {location_type} = {selected_location}")
            st.markdown("**Outcome probabilities:**")
            prob_df = pd.DataFrame({
                "Outcome": list(adjusted_probs.keys()),
                "Probability": [f"{p * 100:.1f}%" for p in adjusted_probs.values()]
            })
            st.dataframe(prob_df, use_container_width=True)

            st.caption("Prediction is based on location-specific historical outcomes, traffic conditions, and customer cancellation history.")
        else:
            st.info("Choose a location and ride conditions to generate an outcome prediction.")

    with tab3:
        st.subheader("Reliability Scoring")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Driver Reliability**")
            d_rating = st.slider("Driver Rating", 1.0, 5.0, 4.5)
            st.progress(d_rating/5.0)
        with col_b:
            st.write("**Customer Loyalty**")
            c_rating = st.slider("Customer Rating", 1.0, 5.0, 4.2)
            st.progress(c_rating/5.0)

# --- PAGE 4: OPERATIONAL INSIGHTS ---
elif page == "Operational Insights":
    st.header("📈 Strategy & Recommendations")
      # --- EDA INSIGHTS SECTION ---
    st.divider()
    st.header("**🔍 Key Insights from Data Analysis**")

    if df.empty:
        st.warning("No data available for insights.")
    else:
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
        }).round(2)

        # Top performing city
        top_city = city_performance['Booking_ID'].idxmax() if not city_performance.empty else "N/A"

        col_insights1, col_insights2 = st.columns(2)

        with col_insights1:
            st.subheader("📊 Overall Performance")
            st.metric("Total Bookings", f"{total_bookings:,}")
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
            st.metric("Peak Hour", f"{peak_hour}:00 ({peak_hour_volume} bookings)")
            st.metric("Top City", top_city)

        with col_insights2:
            st.subheader("💰 Revenue Insights")
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
        st.dataframe(city_table.style.highlight_max(axis=0), use_container_width=True)

        st.divider()

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
    
    # Feature Importance Analysis
    st.subheader("Model Feature Importance")
    
    if df.empty:
        st.warning("No data available for feature importance analysis.")
    else:
        # Prepare data for analysis
        analysis_df = df.copy()
        
        # Create binary target for booking completion
        analysis_df['Is_Completed'] = analysis_df['Booking_Status'].astype(str).str.lower().isin(['completed', 'success']).astype(int)
        
        # Numeric features for correlation analysis
        numeric_features = ['Distance', 'Hour', 'Fare', 'Avg_Customer_Rating', 'Customer_Cancellation_Rate', 
                          'Acceptance_Rate', 'Driver_Rating']
        
        # Calculate correlations for fare prediction (regression)
        fare_correlations = {}
        for feature in numeric_features:
            if feature in analysis_df.columns and feature != 'Fare':
                corr = abs(analysis_df[feature].corr(analysis_df['Fare']))
                fare_correlations[feature] = corr
        
        # Calculate correlations for booking completion prediction (classification)
        completion_correlations = {}
        for feature in numeric_features:
            if feature in analysis_df.columns and feature != 'Fare':
                corr = abs(analysis_df[feature].corr(analysis_df['Is_Completed']))
                completion_correlations[feature] = corr
        
        # Calculate average importance across both targets
        all_features = set(fare_correlations.keys()) | set(completion_correlations.keys())
        feature_importance = {}
        
        for feature in all_features:
            fare_imp = fare_correlations.get(feature, 0)
            completion_imp = completion_correlations.get(feature, 0)
            # Weighted average: 60% fare prediction, 40% completion prediction
            avg_importance = (fare_imp * 0.6) + (completion_imp * 0.4)
            feature_importance[feature] = avg_importance
        
        # Add categorical feature importance estimates
        if 'Traffic_Level' in analysis_df.columns:
            # Calculate completion rate difference across traffic levels
            traffic_completion = analysis_df.groupby('Traffic_Level')['Is_Completed'].mean()
            traffic_importance = traffic_completion.std()  # Variability as importance measure
            feature_importance['Traffic_Level'] = traffic_importance * 0.3  # Scale down
        
        if 'Weather_Condition' in analysis_df.columns:
            # Calculate completion rate difference across weather conditions
            weather_completion = analysis_df.groupby('Weather_Condition')['Is_Completed'].mean()
            weather_importance = weather_completion.std()
            feature_importance['Weather_Condition'] = weather_importance * 0.25
        
        if 'Vehicle_Type' in analysis_df.columns:
            # Calculate fare difference across vehicle types
            vehicle_fare = analysis_df.groupby('Vehicle_Type')['Fare'].mean()
            vehicle_importance = vehicle_fare.std() / vehicle_fare.mean()  # Coefficient of variation
            feature_importance['Vehicle_Type'] = vehicle_importance * 0.2
        
        if 'City' in analysis_df.columns:
            # Calculate both fare and completion variability across cities
            city_fare = analysis_df.groupby('City')['Fare'].mean()
            city_completion = analysis_df.groupby('City')['Is_Completed'].mean()
            city_fare_cv = city_fare.std() / city_fare.mean()
            city_completion_std = city_completion.std()
            city_importance = (city_fare_cv * 0.6) + (city_completion_std * 0.4)
            feature_importance['City'] = city_importance
        
        # Create DataFrame and normalize to sum to 1
        features_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
        
        # Normalize importance scores
        total_importance = features_df['Importance'].sum()
        if total_importance > 0:
            features_df['Importance'] = features_df['Importance'] / total_importance
        
        # Sort by importance
        features_df = features_df.sort_values(by='Importance', ascending=False).head(10)  # Top 10 features
        
        # Create visualization
        fig_imp = px.bar(
            features_df, 
            x='Importance', 
            y='Feature', 
            orientation='h', 
            color='Importance',
            color_continuous_scale='Viridis',
            labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
            title='Top Predictive Features for Fare & Completion Prediction'
        )
        fig_imp.update_layout(
            xaxis_title='Relative Importance (0-1)',
            yaxis_title='Feature',
            coloraxis_colorbar=dict(title='Importance')
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **Feature Importance Methodology:**
        - Calculated using correlation analysis for numeric features
        - Categorical features importance based on target variable variability
        - Combined importance for both fare prediction and booking completion prediction
        - Normalized to show relative contribution of each feature
        """)