import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pymysql
except ImportError:
    print("Error: pymysql is not installed. Installing pymysql...", flush=True)
    os.system(f"{sys.executable} -m pip install pymysql")
    import pymysql

from db.database import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, init_db, fetch_all_records

def main():
    print("=" * 60)
    print("      MYSQL DATABASE SETUP & VERIFICATION SCRIPT")
    print("=" * 60)
    print(f"Target MySQL Host     : {MYSQL_HOST}")
    print(f"Target MySQL Port     : {MYSQL_PORT}")
    print(f"Target MySQL User     : {MYSQL_USER}")
    print(f"Target MySQL Database : {MYSQL_DB}")
    print("=" * 60)

    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            connect_timeout=5
        )
        print("✓ Connected successfully to MySQL Server!")
        
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        conn.commit()
        conn.close()
        print(f"✓ Database `{MYSQL_DB}` created/verified.")

        db_engine = init_db()
        print(f"✓ Table `verification_history` initialized on {db_engine} engine.")

        records, engine = fetch_all_records()
        print(f"✓ Current record count in {engine}: {len(records)}")
        print("\nMySQL database setup is complete and active!")
    except Exception as e:
        print(f"\nCould not connect to MySQL server: {e}")
        print("\nTroubleshooting Tips:")
        print("1. Ensure MySQL (XAMPP / MySQL Workbench / Docker) is running.")
        print("2. Pass credentials as environment variables if non-default:")
        print("   $env:MYSQL_USER='your_username'")
        print("   $env:MYSQL_PASSWORD='your_password'")
        print("   python scripts/setup_mysql.py")

if __name__ == "__main__":
    main()
