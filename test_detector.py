import os
import importlib
import pytest

# Ensure unit tests use the local fallback model
os.environ.pop("WATSON_NLU_APIKEY", None)
os.environ.pop("WATSON_NLU_URL", None)

# Import from either "emotion_app" or fallback to "Final"
try:
    pkg = importlib.import_module("emotion_app")
except ModuleNotFoundError:
    pkg = importlib.import_module("Final")

emotion_detector = getattr(pkg, "emotion_detector")
InvalidTextError = getattr(pkg, "InvalidTextError")


def test_invalid_text_raises():
    with pytest.raises(InvalidTextError):
        emotion_detector("")


def test_fallback_model_scores_exist():
    res = emotion_detector("I am happy and full of joy, a little anxious, and honestly surprised.")
    cores = {"joy", "sadness", "anger", "fear", "disgust", "passion", "surprise"}

    # dominant can be any core or N/A
    assert res.dominant_emotion in cores.union({"N/A"})

    # Each score is within [0, 1]
    for key in cores:
        val = getattr(res, key)
        assert 0.0 <= val <= 1.0
