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

    # Resize for faster detection
    max_size = 300
    scale = 1.0
    if max(orig_h, orig_w) > max_size:
        scale = max_size / max(orig_h, orig_w)
        img = cv2.resize(img, (int(orig_w*scale), int(orig_h*scale)))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_data = []

    # Detect faces
    try:
        detections = detector.detect_faces(img_rgb)
        print(f"[DEBUG] Detections: {detections}")  # Debug output
    except Exception as e:
        print(f"[ERROR] MTCNN detection failed: {e}")
        detections = []

    if not detections:
        return jsonify({"faces": []})

    face_imgs = []
    coords = []

    for det in detections:
        x, y, w, h = det["box"]
        x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
        face = img_rgb[y:y+h, x:x+w]
        if face.size == 0:
            continue
        face_resized = cv2.resize(face, (160,160))
        face_imgs.append(face_resized)
        coords.append((x, y, w, h))

    if not face_imgs:
        return jsonify({"faces": []})

    embeddings = embedder.embeddings(face_imgs)
    embeddings_norm = normalizer.transform(embeddings)

    preds = svc_model.predict(embeddings_norm)
    probs = svc_model.predict_proba(embeddings_norm)

    for i, emb in enumerate(embeddings):
        conf = max(probs[i])
        name = label_encoder.inverse_transform([preds[i]])[0] if conf > 0.6 else "Unknown"
        x, y, w, h = coords[i]
        # Scale back to original image size
        if scale < 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
        faces_data.append({"x": x, "y": y, "w": w, "h": h, "name": name})

    print(f"[DEBUG] Faces returned: {faces_data}")  # Debug output
    return jsonify({"faces": faces_data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
