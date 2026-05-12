# CSV to Table and MySQL insertion
import pandas as pd
from sqlalchemy import text
from Database.DBConnection import get_sqlalchemy_engine

DEFAULT_BATCH_SIZE = 1000  # Reduced from 10000 for safer inserts

CSV_TABLE_METADATA = [
    {
        "csv_filepath": "DataSets/bookings.csv",
        "table_name": "bookings",
        "duplicate_key": "booking_id",
        "boolean_columns": ["is_weekend"],
        "categorical_columns": ["city", "pickup_location", "drop_location", "vehicle_type", "booking_status", "incomplete_ride_reason"],
        "numeric_columns": ["ride_distance_km", "estimated_ride_time_min", "actual_ride_time_min", "hour_of_day", "base_fare", "surge_multiplier", "booking_value"],
    },
    {
        "csv_filepath": "DataSets/customers.csv",
        "table_name": "customers",
        "duplicate_key": "customer_id",
        "boolean_columns": ["customer_cancel_flag"],
        "categorical_columns": ["customer_gender", "customer_city", "preferred_vehicle_type"],
        "numeric_columns": ["customer_age", "customer_signup_days_ago", "total_bookings", "completed_rides", "cancelled_rides", "incomplete_rides", "cancellation_rate", "avg_customer_rating"],
    },
    {
        "csv_filepath": "DataSets/drivers.csv",
        "table_name": "drivers",
        "duplicate_key": "driver_id",
        "boolean_columns": ["driver_delay_flag"],
        "categorical_columns": ["driver_city", "vehicle_type"],
        "numeric_columns": ["driver_age", "driver_experience_years", "total_assigned_rides", "accepted_rides", "incomplete_rides", "delay_count", "acceptance_rate", "delay_rate", "avg_driver_rating", "avg_pickup_delay_min"],
    },
    {
        "csv_filepath": "DataSets/time_features.csv",
        "table_name": "time_features",
        "duplicate_key": "datetime",
        "boolean_columns": ["is_weekend", "is_holiday", "peak_time_flag"],
        "categorical_columns": ["day_of_week", "season"],
        "numeric_columns": ["hour_of_day"],
    },
    {
        "csv_filepath": "DataSets/location_demand.csv",
        "table_name": "location_demand",
        "duplicate_key": None,
        "boolean_columns": [],
        "categorical_columns": ["city", "pickup_location", "vehicle_type"],
        "numeric_columns": ["hour_of_day", "total_requests", "completed_rides", "cancelled_rides", "avg_wait_time_min", "avg_surge_multiplier", "demand_level"],
    },
]

def clear_table(table_name, reset_autoincrement=True):
    """Delete all rows from a table and optionally reset AUTO_INCREMENT."""
    engine = get_sqlalchemy_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {table_name}"))
            if reset_autoincrement:
                conn.execute(text(f"ALTER TABLE {table_name} AUTO_INCREMENT = 1"))
        print(f"Cleared table '{table_name}'{', reset AUTO_INCREMENT' if reset_autoincrement else ''}.")
    except Exception as e:
        print(f"Error clearing table '{table_name}': {e}")
        raise
    finally:
        engine.dispose()

def get_csv_dataframe(file_path):
    """Read data from CSV and return a DataFrame with SQL-friendly nulls."""
    try:
        df = pd.read_csv(file_path)

        # Cast to object so None is preserved and not coerced back to NaN. MySQL connector expects Python None for SQL NULL values.
        df = df.astype(object)
        return df.where(pd.notna(df), None)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()
    
def data_cleaning(df, table_config):
    """Run the standard cleaning pipeline for the given table metadata."""
    df = remove_duplicate_rows(df, table_config.get("duplicate_key"))

    for boolean_column in table_config.get("boolean_columns", []):
        df = normalize_boolean_data(df, boolean_column)

    df = normalize_numeric_columns(df, table_config.get("numeric_columns", []))
    df = normalize_categorical_data(df, table_config.get("categorical_columns", []))
    return df

def remove_duplicate_rows(df, duplicateKey):
    print(f"Checking for duplicate rows based on {duplicateKey}...")
    
    if duplicateKey is None or duplicateKey not in df.columns:
        print(f"Deduplication skipped: column '{duplicateKey}' not found.")
        return df

    # Remove duplicates based on the requested key, keeping the last occurrence. 
    deduplicated_df = df.drop_duplicates(subset=[duplicateKey], keep="last")
    removed_count = len(df) - len(deduplicated_df)

    print(f"Removed {removed_count} duplicate rows based on {duplicateKey} before batch insert.")

    return deduplicated_df

def normalize_numeric_columns(df, numneric_columns): 
    """Normalize the numeric columns"""
    for column in numneric_columns:
        if column not in df.columns:
            print(f"Normalization skipped: column '{column}' not found.")
            continue

        normalized = df[column].astype("string").str.strip().str.replace(r"[^0-9.]", "", regex=True) # Strip whitespace and remove non-numeric characters (e.g. "₹1.5 Cr" → "1.5")
        numeric = pd.to_numeric(normalized, errors="coerce").round(2) # Convert to float and round to 2 decimal places

        change_mask = numeric.notna() # Boolean mask for rows with valid numeric values
        converted_count = int(change_mask.sum())
        df.loc[change_mask, column] = numeric.loc[change_mask] # Store as decimal in the DataFrame
        print(f"Normalized {converted_count} values in '{column}' column to decimal (2dp) before batch insert.")
    return df

def normalize_categorical_data(df, categorical_columns):
    """Normalize categorical columns in the DataFrame before insert."""

    for column in categorical_columns:
        if column not in df.columns:
            continue

        normalized = df[column].astype("string").str.strip() # Remove leading/trailing whitespace and convert to string for consistent processing
        change_mask = normalized.notna() & normalized.ne("") # creates a boolean mask for rows that are non-null and non-empty values for normalization
        normalized_upper = normalized.str.upper() # Standardize to uppercase for consistent capitalization

        normalized_count = int((df.loc[change_mask, column] != normalized_upper.loc[change_mask]).sum())
        df.loc[change_mask, column] = normalized_upper.loc[change_mask] # Update only the rows that have non-null and non-empty values with their normalized uppercase versions
        print(f"Standardized capitalization for {normalized_count} values in column '{column}' before batch insert.")

    return df

def normalize_boolean_data(df, boolean_column):
    """Normalize boolean columns in the DataFrame before insert."""
    
    if boolean_column not in df.columns:
        print(f"Boolean normalization skipped: column '{boolean_column}' not found.")
        return df

    col = df[boolean_column].astype("string").str.strip().str.lower()
    yes_mask = col.eq("yes")
    no_mask = col.eq("no")

    converted_count = int((yes_mask | no_mask).sum())
    df.loc[yes_mask, boolean_column] = True
    df.loc[no_mask, boolean_column] = False
    print(f"Converted {converted_count} Yes/No values in '{boolean_column}' to true/false before batch insert.")

    return df     

def insert_dataframe_to_db(df, table_name, batch_size=DEFAULT_BATCH_SIZE):
    """Insert DataFrame into database with improved error handling and retry logic."""
    max_retries = 3
    retry_count = 0
    engine = None
    
    try:
        print(f"Data batch insert started with batch size {batch_size} into '{table_name}'...")
        print(f"Total rows to insert: {len(df)}")
        
        while retry_count < max_retries:
            try:
                engine = get_sqlalchemy_engine()
                insert_df = df.copy()
                insert_df = insert_df.where(pd.notna(insert_df), None)
                
                # Removed method="multi" for better compatibility with mysql-connector-python
                insert_df.to_sql(
                    name=table_name,
                    con=engine,
                    if_exists="append",
                    index=False,
                    chunksize=batch_size,
                )
                print(f"✓ Data inserted successfully into '{table_name}'. Total rows: {len(insert_df)}")
                return  # Success, exit the function
                
            except Exception as batch_error:
                retry_count += 1
                print(f"Attempt {retry_count}/{max_retries} failed for '{table_name}': {batch_error}")
                
                if engine:
                    engine.dispose()
                
                if retry_count >= max_retries:
                    raise  # Re-raise if all retries exhausted
                else:
                    print(f"Retrying insert for '{table_name}'...")
                    
    except Exception as e:
        print(f"✗ Error inserting data into '{table_name}' after {max_retries} retries: {e}")
        print(f"Table: {table_name}")
        print(f"Rows attempting to insert: {len(df)}")
        raise
    finally:
        if engine:
            try:
                engine.dispose()
            except Exception as cleanup_error:
                print(f"Warning: Error disposing engine: {cleanup_error}")

def process_table_metadata(table_config):
    #Get the DataFrame from the CSV file for the current table
    df = get_csv_dataframe(table_config["csv_filepath"])
   
    """Clean the data and insert it into the provided MySQL table in batches."""
    if df.empty:
        print(f"No data to insert for {table_config['table_name']}.")
        return

    df = data_cleaning(df, table_config)

    insert_dataframe_to_db(df, table_config["table_name"], batch_size=DEFAULT_BATCH_SIZE)


def data_insertion():
    """Main function to clear tables and load data from CSV files."""
    for metadata in CSV_TABLE_METADATA:
        process_table_metadata(metadata)
