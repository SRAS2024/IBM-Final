from emotion_app import emotion_detector, InvalidTextError


def test_invalid_text_raises():
    try:
        emotion_detector("")
        assert False, "Expected InvalidTextError"
    except InvalidTextError:
        assert True


def test_fallback_model_scores_exist():
    # With no credentials, fallback model will run. We only assert presence of keys.
    result = emotion_detector("I am happy and full of joy, nothing to fear.")
    assert result.dominant_emotion in {"joy", "fear", "anger", "disgust", "sadness"}
    assert 0.0 <= result.joy <= 1.0
    assert 0.0 <= result.sadness <= 1.0
    assert 0.0 <= result.anger <= 1.0
    assert 0.0 <= result.fear <= 1.0
    assert 0.0 <= result.disgust <= 1.0
