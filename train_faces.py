import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
import pickle
from mtcnn import MTCNN

DATA_DIR = "Data"
detector = MTCNN()
embedder = FaceNet()

faces, labels = [], []

for person_name in os.listdir(DATA_DIR):
    person_path = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_path): continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for det in detector.detect_faces(img_rgb):
            x, y, w, h = det["box"]
            x, y = max(0,x), max(0,y)
            face = img_rgb[y:y+h, x:x+w]
            if face.size == 0: continue
            faces.append(embedder.embeddings([cv2.resize(face,(160,160))])[0])
            labels.append(person_name)

faces = np.array(faces)
labels = np.array(labels)

normalizer = Normalizer(norm='l2')
faces = normalizer.transform(faces)

label_encoder = LabelEncoder()
labels_enc = label_encoder.fit_transform(labels)

model = SVC(kernel='linear', probability=True)
model.fit(faces, labels_enc)

with open("embeddings.pkl", "wb") as f:
    pickle.dump((model, label_encoder, normalizer), f)

print("✅ Training complete. Embeddings saved to embeddings.pkl")
