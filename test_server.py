import os
import json

# Ensure unit tests use the local fallback model
os.environ.pop("WATSON_NLU_APIKEY", None)
os.environ.pop("WATSON_NLU_URL", None)

from server import app  # noqa: E402


def test_post_emotion_detector_ok():
    client = app.test_client()
    resp = client.post(
        "/emotionDetector",
        data=json.dumps({"text": "I am very happy today."}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    for key in ["anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"]:
        assert key in data


def test_post_emotion_detector_bad_request():
    client = app.test_client()
    resp = client.post(
        "/emotionDetector",
        data=json.dumps({"text": "   "}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
