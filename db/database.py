import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env variables
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Optional MySQL driver imports
PYMYSQL_AVAILABLE = False
try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    pass

# DB Configuration
MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DB = os.environ.get("MYSQL_DB", "deepfake_forensics")

DB_DIR = PROJECT_ROOT / "data" / "db"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_DB_PATH = DB_DIR / "verification_history.db"
JSON_LOG_PATH = LOGS_DIR / "audit_trail.json"


def get_mysql_connection(create_db_if_missing=True):
    """
    Attempts to establish a PyMySQL connection using .env or environment variables.
    """
    if not PYMYSQL_AVAILABLE:
        return None

    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            connect_timeout=3
        )
        if create_db_if_missing:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            conn.commit()
        conn.close()

        db_conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            autocommit=True,
            connect_timeout=3
        )
        return db_conn
    except Exception as e:
        return None


def init_db():
    """
    Initializes MySQL table if MySQL connection succeeds; otherwise initializes SQLite fallback.
    """
    mysql_conn = get_mysql_connection()
    if mysql_conn is not None:
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS verification_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp VARCHAR(64) NOT NULL,
                        request_id VARCHAR(64) NOT NULL,
                        filename VARCHAR(255) NOT NULL,
                        fraud_risk_percent FLOAT NOT NULL,
                        prediction VARCHAR(32) NOT NULL,
                        trigger_modality VARCHAR(255) NOT NULL,
                        analyst_decision VARCHAR(64) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
            mysql_conn.close()
            print("Successfully initialized MySQL Database table: verification_history", flush=True)
            return "MYSQL"
        except Exception as e:
            print(f"MySQL table creation notice: {e}", flush=True)

    # SQLite Fallback Initialization
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                request_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                fraud_risk_percent REAL NOT NULL,
                prediction TEXT NOT NULL,
                trigger_modality TEXT NOT NULL,
                analyst_decision TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        return "SQLITE"
    except Exception as e:
        print(f"SQLite DB init notice: {e}", flush=True)
        return "JSON"


def insert_verification_record(entry):
    """
    Inserts verification decision record into MySQL database, SQLite fallback, and JSON log.
    """
    db_engine = "NONE"
    
    # 1. Try MySQL Insert
    mysql_conn = get_mysql_connection()
    if mysql_conn is not None:
        try:
            with mysql_conn.cursor() as cursor:
                sql = """
                    INSERT INTO verification_history 
                    (timestamp, request_id, filename, fraud_risk_percent, prediction, trigger_modality, analyst_decision)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """
                cursor.execute(sql, (
                    entry.get("timestamp", datetime.now().isoformat()),
                    entry.get("request_id", "N/A"),
                    entry.get("filename", "N/A"),
                    float(entry.get("fraud_risk_percent", 0.0)),
                    entry.get("prediction", "N/A"),
                    entry.get("trigger_modality", "N/A"),
                    entry.get("analyst_decision", "N/A")
                ))
            mysql_conn.close()
            db_engine = "MYSQL"
        except Exception as e:
            print(f"MySQL insert error: {e}", flush=True)

    # 2. Try SQLite Insert (Backup)
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO verification_history 
            (timestamp, request_id, filename, fraud_risk_percent, prediction, trigger_modality, analyst_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.get("timestamp", datetime.now().isoformat()),
            entry.get("request_id", "N/A"),
            entry.get("filename", "N/A"),
            float(entry.get("fraud_risk_percent", 0.0)),
            entry.get("prediction", "N/A"),
            entry.get("trigger_modality", "N/A"),
            entry.get("analyst_decision", "N/A")
        ))
        conn.commit()
        conn.close()
        if db_engine == "NONE":
            db_engine = "SQLITE"
    except Exception as e:
        print(f"SQLite insert error: {e}", flush=True)

    # 3. JSON Log Append (Backup)
    try:
        logs = []
        if JSON_LOG_PATH.exists():
            with open(JSON_LOG_PATH, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except Exception:
                    logs = []
        logs.append(entry)
        with open(JSON_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"JSON write error: {e}", flush=True)

    return db_engine


def fetch_all_records():
    """
    Fetches stored verification records from MySQL database if connected, or SQLite fallback.
    """
    # 1. Fetch from MySQL
    mysql_conn = get_mysql_connection()
    if mysql_conn is not None:
        try:
            records = []
            with mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT * FROM verification_history ORDER BY id DESC;")
                records = cursor.fetchall()
            mysql_conn.close()
            return records, "MYSQL"
        except Exception as e:
            print(f"MySQL fetch error: {e}", flush=True)

    # 2. Fetch from SQLite
    if SQLITE_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM verification_history ORDER BY id DESC;")
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            conn.close()
            return records, "SQLITE"
        except Exception as e:
            print(f"SQLite fetch error: {e}", flush=True)

    # 3. Fetch from JSON
    if JSON_LOG_PATH.exists():
        try:
            with open(JSON_LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
                return logs, "JSON"
        except Exception:
            pass

    return [], "NONE"
