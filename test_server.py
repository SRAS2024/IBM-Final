import os
import json
import pytest
from server import app

# Force local fallback in tests
os.environ.pop("WATSON_NLU_APIKEY", None)
os.environ.pop("WATSON_NLU_URL", None)


def test_post_emotion_detector_ok():
    client = app.test_client()
    resp = client.post(
        "/emotionDetector",
        data=json.dumps({"text": "I am thrilled, a bit anxious, and honestly surprised!"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    # Core scores present
    for key in [
        "anger",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "passion",
        "surprise",
        "dominant_emotion",
        "blended_emotion",
        "emotion",
        "confidence",
        "mixture",
        "components",
        "emoji",
    ]:
        assert key in data, f"missing key: {key}"

    # Ranges and shapes
    for k in ["anger", "disgust", "fear", "joy", "sadness", "passion", "surprise"]:
        assert 0.0 <= float(data[k]) <= 1.0

    assert isinstance(data["dominant_emotion"], str)
    assert isinstance(data["blended_emotion"], str)
    assert isinstance(data["emotion"], str)
    assert 0.0 <= float(data["confidence"]) <= 1.0

    mix = data["mixture"]
    for k in ["anger", "disgust", "fear", "joy", "sadness", "passion", "surprise"]:
        assert k in mix
        assert 0.0 <= float(mix[k]) <= 1.0

    comps = data["components"]
    assert isinstance(comps, list)
    assert len(comps) >= 1
    assert isinstance(comps[0], list) or isinstance(comps[0], tuple)

    emoji = data["emoji"]
    assert isinstance(emoji, list)
    assert 1 <= len(emoji) <= 2
    assert all(isinstance(e, str) for e in emoji)


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
