# app.py (UPDATED)
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet

app = Flask(__name__)

# Load trained embeddings, classifier, and normalizer
with open("embeddings.pkl", "rb") as f:
    svc_model, label_encoder, normalizer = pickle.load(f)

detector = MTCNN()
embedder = FaceNet()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")
    if not file:
        return jsonify({"faces": []})

    # Read uploaded image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"faces": []})

    orig_h, orig_w = img.shape[:2]

    # Resize for faster detection (maintain scaling info)
    max_size = 300
    scale = 1.0
    if max(orig_h, orig_w) > max_size:
        scale = max_size / max(orig_h, orig_w)
        resized_w = int(orig_w * scale)
        resized_h = int(orig_h * scale)
        img = cv2.resize(img, (resized_w, resized_h))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_data = []

    # Detect faces
    try:
        detections = detector.detect_faces(img_rgb)
        print(f"[DEBUG] Detections: {detections}")
    except Exception as e:
        print(f"[ERROR] MTCNN detection failed: {e}")
        detections = []

    if not detections:
        return jsonify({"faces": []})

    face_imgs = []
    coords = []

    for det in detections:
        # MTCNN returns 'box' as [x, y, w, h]
        x, y, w, h = det.get("box", (0, 0, 0, 0))
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Clip to image
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
            print(f"[WARN] face resize failed: {e}")
            continue

        face_imgs.append(face_resized)
        coords.append((x1, y1, x2 - x1, y2 - y1))

    if not face_imgs:
        return jsonify({"faces": []})

    # Get embeddings (ensure we have a numpy array of shape (N, dim))
    try:
        embeddings = embedder.embeddings(face_imgs)
        embeddings = np.asarray(embeddings)
    except Exception as e:
        print("[ERROR] FaceNet embedding failed:", e)
        return jsonify({"faces": []})

    # Normalize embeddings (normalizer expects 2D array)
    try:
        embeddings_norm = normalizer.transform(embeddings)
    except Exception as e:
        print("[ERROR] Normalizer transform failed:", e)
        return jsonify({"faces": []})

    # Predict labels and probabilities
    try:
        preds = svc_model.predict(embeddings_norm)
        probs = svc_model.predict_proba(embeddings_norm)
    except Exception as e:
        print("[ERROR] Classifier prediction failed:", e)
        return jsonify({"faces": []})

    for i in range(len(embeddings)):
        conf = float(np.max(probs[i])) if probs is not None else 0.0
        predicted_label = preds[i]

        # get readable name (safeguard)
        try:
            name = label_encoder.inverse_transform([predicted_label])[0] if conf > 0.6 else "Unknown"
        except Exception:
            # If label decoding fails, fall back to Unknown
            name = "Unknown"

        x, y, w, h = coords[i]
        # Scale back to original image size if we resized earlier
        if scale < 1.0:
            # coords are on resized image; convert to original coordinates
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        faces_data.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "name": str(name),
            "confidence": float(conf)
        })

    print(f"[DEBUG] Faces returned: {faces_data}")
    return jsonify({"faces": faces_data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
