"""Flask server for the Emotion Detector project."""
from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request
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

INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Emotion Detector</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 2rem; }
      .card { max-width: 720px; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
      textarea { width: 100%; min-height: 120px; }
      pre { background: #f8f8f8; padding: 0.75rem; border-radius: 6px; overflow: auto; }
      button { padding: 0.5rem 1rem; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Emotion Detector</h1>
      <p>Enter text and submit. The API returns emotion scores and the dominant emotion.</p>
      <textarea id="text" placeholder="Type here..."></textarea>
      <br><br>
      <button onclick="detect()">Analyze</button>
      <h3>Result</h3>
      <pre id="out">{}</pre>
    </div>
    <script>
      async function detect() {
        const text = document.getElementById('text').value;
        const resp = await fetch('/emotionDetector', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const data = await resp.json();
        document.getElementById('out').textContent = JSON.stringify(data, null, 2);
      }
    </script>
  </body>
</html>
"""


@app.get("/")
def index() -> Any:
    return render_template_string(INDEX_HTML)


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
    except Exception as exc:  # unexpected
        return jsonify({"error": f"Internal server error: {exc}"}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port)
