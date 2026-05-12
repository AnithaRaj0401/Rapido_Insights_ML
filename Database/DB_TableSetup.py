from sqlalchemy import text
from Database.DBConnection import DB_NAME, get_sqlalchemy_engine

DB_SQL_TEXT = f"""
CREATE DATABASE IF NOT EXISTS `{DB_NAME}`;
"""

DROP_CREATE_TABLE_SQL = f"""
USE `{DB_NAME}`;

DROP TABLE IF EXISTS Bookings;
DROP TABLE IF EXISTS Customers;
DROP TABLE IF EXISTS Drivers;
DROP TABLE IF EXISTS Location_Demand;
DROP TABLE IF EXISTS Time_Features;

CREATE TABLE Bookings (
    Booking_ID VARCHAR(10) PRIMARY KEY NOT NULL,
    Booking_Date DATE NOT NULL,
    Booking_Time TIME NOT NULL,
    Day_of_Week VARCHAR(10) NOT NULL,
    Is_Weekend BOOLEAN NOT NULL,
    Hour_of_Day INT NOT NULL,
    City VARCHAR(100) NOT NULL,
    Pickup_Location VARCHAR(200) NOT NULL,
    Drop_Location VARCHAR(200) NOT NULL,
    Vehicle_Type VARCHAR(50) NOT NULL,
    Ride_Distance_KM DECIMAL(10,2) NOT NULL,
    Estimated_Ride_Time_Min INT NOT NULL,
    Actual_Ride_Time_Min INT,
    Traffic_Level VARCHAR(50),
    Weather_Condition VARCHAR(50),
    Base_Fare DECIMAL(10,2) NOT NULL,
    Surge_Multiplier DECIMAL(3,2) DEFAULT 1.00,
    Booking_Value DECIMAL(10,2) NOT NULL,
    Booking_Status VARCHAR(50) NOT NULL,
    Incomplete_Ride_Reason TEXT,
    Customer_ID VARCHAR(10) NOT NULL,
    Driver_ID VARCHAR(10) ,

    INDEX idx_bookings_customer_id (Customer_ID),
    INDEX idx_bookings_driver_id (Driver_ID)
);

CREATE TABLE Customers (
   Customer_ID VARCHAR(10) PRIMARY KEY NOT NULL,
   Customer_Gender VARCHAR(10) NOT NULL,
   Customer_Age INT NOT NULL,
   Customer_City VARCHAR(100) NOT NULL,
   Customer_Signup_Days_Ago INT NOT NULL,
   Preferred_Vehicle_Type VARCHAR(50),
   Total_Bookings INT DEFAULT 0,
   Completed_Rides INT DEFAULT 0,
   Cancelled_Rides INT DEFAULT 0,
   Incomplete_Rides INT DEFAULT 0,
   Cancellation_Rate DECIMAL(5,2) DEFAULT 0.00,
   Avg_Customer_Rating DECIMAL(3,2) DEFAULT 0.00,
   Customer_Cancel_Flag BOOLEAN DEFAULT FALSE
);

CREATE TABLE Drivers (
   Driver_ID VARCHAR(10) PRIMARY KEY NOT NULL,
   Driver_Age INT NOT NULL,
   Driver_City VARCHAR(100) NOT NULL,
   Vehicle_Type VARCHAR(50) NOT NULL,
   Driver_Experience_Years INT NOT NULL,
   Total_Assigned_Rides INT DEFAULT 0,
   Accepted_Rides INT DEFAULT 0,
   Incomplete_Rides INT DEFAULT 0,
   Delay_Count INT DEFAULT 0,
   Acceptance_Rate DECIMAL(5,2) DEFAULT 0.00,
   Delay_Rate DECIMAL(5,2) DEFAULT 0.00,
   Avg_Driver_Rating DECIMAL(3,2) DEFAULT 0.00,
   Avg_Pickup_Delay_Min DECIMAL(5,2) DEFAULT 0.00,
   Driver_Delay_Flag BOOLEAN DEFAULT FALSE
);

CREATE TABLE Location_Demand (
   City VARCHAR(100) NOT NULL,
   Pickup_Location VARCHAR(200) NOT NULL,
   Hour_of_Day INT NOT NULL,
   Vehicle_Type VARCHAR(50) NOT NULL,
   Total_Requests INT DEFAULT 0,
   Completed_Rides INT DEFAULT 0,
   Cancelled_Rides INT DEFAULT 0,
   Avg_Wait_Time_Min DECIMAL(5,2) DEFAULT 0.00,
   Avg_Surge_Multiplier DECIMAL(3,2) DEFAULT 1.00,
   Demand_Level VARCHAR(50) NOT NULL
);

CREATE TABLE Time_Features (
    DateTime DATETIME NOT NULL,
    Hour_of_Day INT NOT NULL,
    Day_of_Week VARCHAR(10) NOT NULL,
    Is_Weekend BOOLEAN NOT NULL,
    Is_Holiday BOOLEAN NOT NULL,
    Peak_Time_Flag BOOLEAN NOT NULL,
    Season VARCHAR(20) NOT NULL
);
"""

def setup_database_and_tables():
    """Set up the database and tables for housing data"""
    server_engine = get_sqlalchemy_engine(database=None)
    with server_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
    print (f"Database '{DB_NAME}' created successfully.")

    database_engine = get_sqlalchemy_engine()
    with database_engine.begin() as connection:
        statements = [stmt.strip() for stmt in DROP_CREATE_TABLE_SQL.split(";") if stmt.strip()]
        for statement in statements:
            connection.execute(text(statement))

    print("Table created successfully.")

    server_engine.dispose()
    database_engine.dispose()