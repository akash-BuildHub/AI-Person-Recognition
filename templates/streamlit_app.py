import streamlit as st
import cv2
import face_recognition
import pickle
import numpy as np
from PIL import Image

# -------------------
# Load encodings
# -------------------
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

st.title("🤖 AI Person Recognition")

mode = st.sidebar.selectbox("Select Mode", ["Upload Image", "Upload Video"])

def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        results.append((name, (left, top, right, bottom)))
    return results

# -------------------
# Image Upload
# -------------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        results = recognize_faces(image)

        for name, (l, t, r, b) in results:
            color = (0,255,0) if name != "Unknown" else (255,0,0)
            cv2.rectangle(image, (l,t), (r,b), color, 2)
            cv2.putText(image, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        st.image(image, channels="RGB")

# -------------------
# Video Upload
# -------------------
elif mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4","avi","mov"])
    if uploaded_file:
        tfile = f"temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = recognize_faces(frame)

            for name, (l, t, r, b) in results:
                color = (0,255,0) if name != "Unknown" else (255,0,0)
                cv2.rectangle(frame, (l,t), (r,b), color, 2)
                cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            stframe.image(frame, channels="BGR")

        cap.release()
