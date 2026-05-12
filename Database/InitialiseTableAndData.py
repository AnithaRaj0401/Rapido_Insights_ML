from Database import DB_TableSetup as db_setup
from Database import DataInsert as data_insert

try:
    # Set up the database and tables
    print("Setting up database and tables...")
    db_setup.setup_database_and_tables()
    print("---------------------------------------------")

    # Insert data into the respective table
    data_insert.data_insertion()
    print("---------------------------------------------")
except Exception as e:
    print(f"Data initialization failed - An error occurred: {e}")