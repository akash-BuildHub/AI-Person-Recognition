# app.py (Render Deployment Ready)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os

app = Flask(__name__, template_folder="templates")
CORS(app)  # Allow frontend JS requests

# ---------------- Load trained models ----------------
with open("embeddings.pkl", "rb") as f:
    svc_model, label_encoder, normalizer = pickle.load(f)

detector = MTCNN()
embedder = FaceNet()

# ---------------- Home route ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- Recognition route ----------------
@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")
    if not file:
        return jsonify({"faces": []})

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"faces": []})

    orig_h, orig_w = img.shape[:2]
    max_size = 300
    scale = 1.0
    if max(orig_h, orig_w) > max_size:
        scale = max_size / max(orig_h, orig_w)
        img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_data = []

    try:
        detections = detector.detect_faces(img_rgb)
    except Exception as e:
        print(f"[ERROR] MTCNN detection failed: {e}")
        detections = []

    if not detections:
        return jsonify({"faces": []})

    face_imgs = []
    coords = []

    for det in detections:
        x, y, w, h = det.get("box", (0, 0, 0, 0))
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_rgb.shape[1], x1 + w), min(img_rgb.shape[0], y1 + h)
        if x2 <= x1 or y2 <= y1:
            continue

        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0:
            continue

        try:
            face_resized = cv2.resize(face, (160, 160))
        except Exception as e:
            print(f"[WARN] Face resize failed: {e}")
            continue

        face_imgs.append(face_resized)
        coords.append((x1, y1, x2 - x1, y2 - y1))

    if not face_imgs:
        return jsonify({"faces": []})

    try:
        embeddings = embedder.embeddings(face_imgs)
        embeddings = np.asarray(embeddings)
        embeddings_norm = normalizer.transform(embeddings)
        preds = svc_model.predict(embeddings_norm)
        probs = svc_model.predict_proba(embeddings_norm)
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return jsonify({"faces": []})

    for i, emb in enumerate(embeddings):
        conf = float(np.max(probs[i])) if probs is not None else 0.0
        predicted_label = preds[i]

        try:
            name = label_encoder.inverse_transform([predicted_label])[0] if conf > 0.6 else "Unknown"
        except Exception:
            name = "Unknown"

        x, y, w, h = coords[i]
        if scale < 1.0:
            x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

        faces_data.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "name": name,
            "confidence": round(conf, 2)
        })

    return jsonify({"faces": faces_data})

if __name__ == "__main__":
    # Use Render's PORT environment variable, default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
