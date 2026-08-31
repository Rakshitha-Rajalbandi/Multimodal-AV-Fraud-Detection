# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from pathlib import Path
# import uuid

# from pipeline import analyze_video


# app = Flask(__name__)
# CORS(app)

# UPLOAD_DIR = Path("uploads")
# UPLOAD_DIR.mkdir(exist_ok=True)


# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "Multimodal Audio-Video Fraud Detection API",
#         "status": "running"
#     })


# @app.route("/predict", methods=["POST"])
# def predict():

#     if "video" not in request.files:
#         return jsonify({
#             "error": "No video file uploaded"
#         }), 400

#     video = request.files["video"]

#     if video.filename == "":
#         return jsonify({
#             "error": "No filename provided"
#         }), 400

#     extension = Path(video.filename).suffix.lower()

#     if extension not in [".mp4", ".avi", ".mov", ".mkv"]:
#         return jsonify({
#             "error": "Unsupported video format"
#         }), 400

#     filename = f"{uuid.uuid4().hex}{extension}"
#     video_path = UPLOAD_DIR / filename

#     video.save(video_path)

#     try:
#         result = analyze_video(str(video_path))

#         return jsonify(result)

#     except Exception as e:

#         return jsonify({
#             "error": str(e)
#         }), 500

#     finally:

#         if video_path.exists():
#             video_path.unlink()


# if __name__ == "__main__":
#     app.run(
#         host="127.0.0.1",
#         port=5000,
#         debug=True
#     )




from pathlib import Path
import uuid
import shutil

from flask import Flask, request, jsonify

from pipeline import analyze_video


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

WORK_DIR = PROJECT_ROOT / "data" / "inference"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WORK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Multimodal Audio-Video Deepfake Detection API",
        "status": "running"
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })


# ============================================================
# ANALYZE VIDEO
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if "video" not in request.files:

        return jsonify({
            "error": "No video file provided."
        }), 400

    uploaded_file = request.files["video"]

    if uploaded_file.filename == "":

        return jsonify({
            "error": "No video selected."
        }), 400

    # --------------------------------------------------------
    # Create unique request ID
    # --------------------------------------------------------

    request_id = uuid.uuid4().hex

    request_dir = (
        WORK_DIR /
        request_id
    )

    request_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    original_name = Path(
        uploaded_file.filename
    ).name

    video_path = (
        request_dir /
        original_name
    )

    uploaded_file.save(
        str(video_path)
    )

    print()
    print("=" * 60)
    print("NEW VIDEO ANALYSIS")
    print("=" * 60)

    print(
        "Request ID:",
        request_id
    )

    print(
        "Video:",
        original_name
    )

    try:

        # ----------------------------------------------------
        # Run complete multimodal pipeline
        # ----------------------------------------------------

        results = analyze_video(
            video_path,
            request_dir
        )

        # ----------------------------------------------------
        # Add request information
        # ----------------------------------------------------

        response = {

            "request_id":
                request_id,

            "filename":
                original_name,

            "status":
                "success",

            "results":
                results
        }

        print()
        print("=" * 60)
        print("ANALYSIS COMPLETED")
        print("=" * 60)

        print(
            "Prediction:",
            results["fusion"]["prediction"]
        )

        print(
            "Fraud risk:",
            f"{results['fusion']['fraud_risk_percent']:.2f}%"
        )

        print("=" * 60)

        return jsonify(response)

    except Exception as e:

        print()
        print("ERROR:")
        print(e)

        return jsonify({

            "request_id":
                request_id,

            "status":
                "error",

            "error":
                str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MULTIMODAL DEEPFAKE DETECTION API")
    print("=" * 60)

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "Health: http://127.0.0.1:5000/health"
    )

    print(
        "Prediction endpoint:"
    )

    print(
        "POST http://127.0.0.1:5000/predict"
    )

    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )