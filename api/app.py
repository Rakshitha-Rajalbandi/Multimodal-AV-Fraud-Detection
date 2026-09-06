import os
import sys
import uuid
import json
import shutil
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.pipeline import analyze_video
from db.database import init_db, insert_verification_record, fetch_all_records

# Directories
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
WORK_DIR = PROJECT_ROOT / "data" / "inference"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Database (MySQL / SQLite)
db_status = init_db()

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

# Demo User Store
VALID_USERS = {
    "admin": "admin123",
    "analyst": "password123",
    "forensic": "forensic2026"
}


@app.route("/", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Serves the frontend React/HTML dashboard."""
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api", methods=["GET"])
def api_info():
    """JSON API information endpoint."""
    return jsonify({
        "system": "Multimodal Audio-Video Fraud & Deepfake Detection API",
        "status": "running",
        "database_engine": db_status,
        "coe_focus": "AI Forensic Security & Identity Verification",
        "endpoints": {
            "health": "/health",
            "login": "POST /api/login",
            "predict": "POST /predict",
            "record_decision": "POST /record_decision",
            "audit_trail": "GET /audit_trail",
            "dashboard": "/dashboard"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "deepfake-detector-api", "database": db_status})


@app.route("/api/login", methods=["POST"])
def login():
    """
    User authentication endpoint.
    """
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password required."}), 400

    if username in VALID_USERS and VALID_USERS[username] == password:
        return jsonify({
            "status": "success",
            "user": {
                "username": username,
                "role": "Administrator" if username == "admin" else "Forensic Analyst",
                "token": f"token_{uuid.uuid4().hex}"
            }
        })
    else:
        return jsonify({"error": "Invalid username or password."}), 401


@app.route("/record_decision", methods=["POST"])
def record_decision():
    """
    Stores analyst verification decision in MySQL database / SQLite persistent table.
    """
    data = request.get_json(silent=True) or {}
    decision = data.get("decision", "")
    request_id = data.get("request_id", "N/A")
    filename = data.get("filename", "N/A")
    fraud_risk_percent = data.get("fraud_risk_percent", 0.0)
    prediction = data.get("prediction", "N/A")
    trigger_modality = data.get("trigger_modality", "N/A")

    if not decision:
        return jsonify({"error": "No decision provided."}), 400

    entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "filename": filename,
        "fraud_risk_percent": fraud_risk_percent,
        "prediction": prediction,
        "trigger_modality": trigger_modality,
        "analyst_decision": decision
    }

    try:
        active_db = insert_verification_record(entry)
        records, _ = fetch_all_records()
        return jsonify({
            "status": "recorded",
            "database_used": active_db,
            "audit_entry": entry,
            "total_audit_records": len(records)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audit_trail", methods=["GET"])
@app.route("/api/history", methods=["GET"])
def get_audit_trail():
    """
    Returns stored audit trail log records from MySQL / SQLite database.
    """
    records, engine_used = fetch_all_records()
    return jsonify({
        "audit_trail": records,
        "database_engine": engine_used,
        "count": len(records)
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main endpoint: Uploads video file and runs complete multimodal deepfake detection.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided in request."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return jsonify({"error": f"Unsupported video extension: {ext}"}), 400

    request_id = uuid.uuid4().hex
    request_dir = WORK_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    video_path = request_dir / f"upload{ext}"
    file.save(str(video_path))

    try:
        results = analyze_video(video_path, request_dir)
        return jsonify({
            "request_id": request_id,
            "filename": file.filename,
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("STARTING MULTIMODAL AUDIO-VIDEO DEEPFAKE API SERVER", flush=True)
    print(f"Database Engine Initialized: {db_status}", flush=True)
    print("Dashboard available at: http://127.0.0.1:5000/", flush=True)
    print("=" * 60, flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)