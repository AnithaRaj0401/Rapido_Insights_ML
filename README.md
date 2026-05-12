# Rapido Insights ML

A Streamlit-based analytics dashboard for ride-hailing insights and predictions, built around historical bookings, customer, and driver data.

## Project Overview

This project provides:
- Data ingestion from a MySQL database
- Exploratory Data Analysis (EDA) with visualizations
- Rule-based predictive logic for ride outcome, fare, cancellation risk, and driver delay risk
- Operational insights and recommendations for mobility operations

## Key Features

- **Project Overview**: view raw booking data and summary statistics
- **Exploratory Data Analysis**: city performance, price vs distance, cancellation heatmaps, driver/customer metrics, and weather/traffic impact
- **Predictions**:
  - Ride outcome prediction based on city, traffic, weather, customer history, and vehicle type
  - Estimated fare calculation using distance, city, traffic, weather, surge, and vehicle type
  - Customer cancellation risk scoring using historical cancellations, ratings, peak-time behavior, pricing sensitivity, and loyalty
  - Driver delay risk scoring based on past delay count, traffic exposure, acceptance rate, and reliability
- **Operational Insights**: recommendations for driver allocation, pricing, and reliability improvement

## Prerequisites
1. Python 3.9+
2. MySQL Server running locally
3. Python packages:
   - pandas
   - mysql-connector-python
   - streamlit
   - plotly
   - numpy
   - scikit-learn
   - statsmodels
   - matplotlib
   - seaborn
   - SQLAlchemy
   - requests

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data Source

The app loads data from MySQL using `Database.DBConnection.get_sqlalchemy_engine()`.
The main query joins `bookings`, `customers`, and `drivers` tables to combine booking, customer, and driver information.

## Notes

- The prediction logic is currently rule-based and uses heuristic adjustments rather than a trained ML model.
- The application assumes the MySQL database schema contains the expected columns used in the query.
- The current repository remote push may require valid GitHub authentication or correct remote URL configuration.

## Project Structure

Rapido_Insights_ML/
├── main.py                        # Streamlit application entry point
├── requirements.txt               # Python dependencies
├── DataSets/
│   ├── bookings.csv               # Booking-level historical ride data
│   ├── customers.csv              # Customer profile and cancellation metrics
│   ├── drivers.csv                # Driver performance and rating data
│   ├── location_demand.csv        # Location demand patterns and metrics
│   └── time_features.csv          # Time-based features for rides
└── Database/
    ├── DBConnection.py            # MySQL connection configuration
    ├── DB_TableSetup.py           # Database and table creation SQL
    ├── DataCleaning.py            # SQL-based data cleaning routines
    ├── DataInsert.py              # CSV-to-database ingestion
    └── InitialiseTableAndData.py  # One-time setup runner

## License

This project does not currently specify a license. Add a `LICENSE` file if needed.
