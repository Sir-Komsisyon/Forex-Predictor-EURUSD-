from sqlalchemy import create_engine

# ---------- MySQL Configuration ----------
DB_USER = "root"               # <-- Change if your MySQL username is different
DB_PASSWORD = "putpasswordhere" # <-- Your MySQL password
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "eurusd_predictor"


def get_engine():
    """
    Returns a SQLAlchemy engine connected to MySQL.
    """
    connection_string = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    return create_engine(
        connection_string,
        pool_pre_ping=True
    )
