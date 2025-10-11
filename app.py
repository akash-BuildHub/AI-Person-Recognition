import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.preprocessing import Normalizer

# ---------------- Configuration ----------------
CONFIDENCE_THRESHOLD = 0.75
DETECTION_SCALE = 0.5
# ------------------------------------------------

app = Flask(__name__)

# ---- Load Trained Models ----
with open("embeddings.pkl", "rb") as f:
    svc_model, label_encoder, normalizer = pickle.load(f)

embedder = FaceNet()
detector = MTCNN()

def get_faces_and_embeddings(frame):
    faces_data = []
    try:
        small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        results = detector.detect_faces(small_frame)
        if len(results) == 0:
            return []

        faces_to_embed = []
        coords = []
        for res in results:
            x, y, w, h = res["box"]
            x, y = max(0, int(x / DETECTION_SCALE)), max(0, int(y / DETECTION_SCALE))
            w, h = int(w / DETECTION_SCALE), int(h / DETECTION_SCALE)
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face_resized = cv2.resize(face, (160, 160))
            faces_to_embed.append(face_resized)
            coords.append((x, y, w, h))

        if not faces_to_embed:
            return []

        embs = embedder.embeddings(faces_to_embed)
        embs_norm = normalizer.transform(embs)
        preds = svc_model.predict(embs_norm)
        probs = svc_model.predict_proba(embs_norm)

        for i, (x, y, w, h) in enumerate(coords):
            conf = float(np.max(probs[i]))
            name = "Unknown"
            if conf >= CONFIDENCE_THRESHOLD:
                name = label_encoder.inverse_transform([preds[i]])[0]
            faces_data.append({"name": name, "x": x, "y": y, "w": w, "h": h})

    except Exception as e:
        print("Detection error:", e)

    return faces_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"faces": []})
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = get_faces_and_embeddings(frame_rgb)
    return jsonify({"faces": faces})

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
