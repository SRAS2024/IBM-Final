# Emotion Detector with Flask and Watson NLU

This app exposes an API and a tiny web page that detects primary emotions in text. It uses IBM Watson Natural Language Understanding when credentials are present. It falls back to a simple local heuristic for offline development.

## Quick start

1. Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the env template and set credentials if you have them:

```bash
cp .env.example .env
# fill WATSON_NLU_APIKEY and WATSON_NLU_URL
```

5. Run the server:

```bash
python server.py
```

Open http://127.0.0.1:5000 to use the simple form. The API route is `POST /emotionDetector` with JSON body `{ "text": "your text" }`.

## Tests

```bash
pytest -q
```

## Static code analysis

You can run either tool:

```bash
flake8
pylint server.py emotion_app
```

## Packaging notes

The `emotion_app` directory is a Python package. Public functions are exported in `emotion_app/__init__.py`.
