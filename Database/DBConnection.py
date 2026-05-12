import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Defaults match MySQL Workbench "Local instance MySQL80"
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3307"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_NAME = os.getenv("DB_NAME", "rapido_db")

def get_sqlalchemy_engine(database=DB_NAME):
    """Create a SQLAlchemy engine for MySQL using mysql-connector-python."""
    url_kwargs = {
        "drivername": "mysql+mysqlconnector",
        "username": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
    }
    if database:
        url_kwargs["database"] = database

    connection_url = URL.create(**url_kwargs)
    return create_engine(connection_url, pool_pre_ping=True)
