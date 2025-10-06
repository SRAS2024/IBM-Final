"""Flask server for the Emotion Detector project."""
from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv

from emotion_app import (
    emotion_detector,
    format_emotions,
    InvalidTextError,
    ServiceUnavailableError,
)

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


@app.get("/")
def index() -> Any:
    return send_file("emoindex.html")


@app.get("/health")
def health() -> tuple[str, int]:
    return "ok", 200


@app.post("/emotionDetector")
def detect() -> tuple[Dict[str, Any], int]:
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    try:
        result = emotion_detector(text)
        formatted = format_emotions(result)
        return jsonify(formatted), 200
    except InvalidTextError as exc:
        return jsonify({"error": str(exc)}), 400
    except ServiceUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Internal server error: {exc}"}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port)
