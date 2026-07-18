from db_config import get_engine

try:
    engine = get_engine()

    with engine.connect() as conn:
        print("✅ Connected to MySQL!")

except Exception as e:
    print("❌ Connection Failed")
    print(e)
