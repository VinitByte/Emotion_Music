import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_PATH = "emotion_model.h5"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

_model = None
def get_model():
    global _model
    if _model is None:                      # load once, on first request
        _model = load_model(MODEL_PATH, compile=False)
    return _model

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return "ok"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    img_b64 = data.get("image", "")
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]

    try:
        arr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("bad image")
    except Exception:
        return jsonify(error="invalid image"), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify(face=False)

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
    model = get_model()
    _, ih, iw, ch = model.input_shape

    face = cv2.resize(gray[y:y + h, x:x + w], (iw, ih)) / 255.0
    face = np.stack([face] * 3, axis=-1) if ch == 3 else face.reshape(ih, iw, 1)
    face = face.reshape(1, ih, iw, ch).astype("float32")

    preds = model.predict(face, verbose=0)[0]
    idx = int(np.argmax(preds))
    return jsonify(face=True, emotion=EMOTIONS[idx], confidence=float(preds[idx]))

if __name__ == "__main__":
    app.run(debug=True)
