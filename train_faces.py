import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
import pickle
from mtcnn import MTCNN

# Path to your dataset
DATA_DIR = "data"

# Initialize MTCNN and FaceNet
detector = MTCNN()
embedder = FaceNet()

faces = []
labels = []

# Loop through each person folder
for person_name in os.listdir(DATA_DIR):
    person_path = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(img_rgb)

        for det in detections:
            x, y, w, h = det["box"]
            x, y = max(0, x), max(0, y)
            face = img_rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue

            # Resize face for FaceNet
            face_resized = cv2.resize(face, (160,160))
            emb = embedder.embeddings([face_resized])[0]
            faces.append(emb)
            labels.append(person_name)

faces = np.array(faces)
labels = np.array(labels)

# Normalize embeddings
in_encoder = Normalizer(norm='l2')
faces = in_encoder.transform(faces)

# Label encode targets
out_encoder = LabelEncoder()
out_labels = out_encoder.fit_transform(labels)

# Fit SVC model
model = SVC(kernel='linear', probability=True)
model.fit(faces, out_labels)

# Save model, label encoder, and normalizer
with open("embeddings.pkl", "wb") as f:
    pickle.dump((model, out_encoder, in_encoder), f)

print("Training complete. Embeddings saved to embeddings.pkl")
