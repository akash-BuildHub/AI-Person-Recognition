# streamlit_app.py (UPDATED)
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tempfile
import cv2
import numpy as np

st.set_page_config(page_title="AI Person Recognition", layout="wide")
st.title("AI Person Recognition")

option = st.radio("Select input type:", ("Webcam Snapshot", "Upload Image", "Upload Video"))

BACKEND_URL = "http://127.0.0.1:5000/recognize"

# ---------------- Helper: Send image to backend ----------------
def recognize_image(image_bytes):
    files = {"image": image_bytes}
    try:
        response = requests.post(BACKEND_URL, files=files)
        response.raise_for_status()
        data = response.json()
        return data.get("faces", [])
    except Exception as e:
        st.error(f"Error contacting backend: {e}")
        return []

# ---------------- Helper: Draw bounding boxes ----------------
def draw_boxes(image, faces, is_video=False):
    img_np = np.array(image.convert("RGB"))

    for face in faces:
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        name = face.get("name", "Unknown")
        conf = face.get("confidence", 0.0)

        # Apply strict rule: below threshold => Unknown
        if conf < 0.6:
            name = "Unknown"

        # For display: show name + confidence
        label_text = f"{name} ({conf:.2f})" if conf else name

        # Color coding: green = known, red = unknown / Person N
        if not is_video and name != "Unknown":
            color = (0, 255, 0)
        elif is_video and not name.startswith("Person") and name != "Unknown":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.rectangle(img_np, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img_np,
            label_text,
            (x, y - 10 if y > 20 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return Image.fromarray(img_np)

# ---------------- Webcam Snapshot ----------------
if option == "Webcam Snapshot":
    st.warning("Webcam streaming not fully supported. Use single snapshot below.")
    img_file = st.camera_input("Take a picture")
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Captured Image", use_column_width=True)

        with BytesIO() as buf:
            image.save(buf, format="JPEG")
            faces = recognize_image(buf.getvalue())

        result_img = draw_boxes(image, faces, is_video=False)
        st.image(result_img, caption="Detected Faces", use_column_width=True)

# ---------------- Upload Image ----------------
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with BytesIO() as buf:
            image.save(buf, format="JPEG")
            faces = recognize_image(buf.getvalue())

        # Force Unknown if confidence < threshold
        for f in faces:
            if f.get("confidence", 0) < 0.6:
                f["name"] = "Unknown"

        result_img = draw_boxes(image, faces, is_video=False)
        st.image(result_img, caption="Detected Faces", use_column_width=True)

# ---------------- Upload Video ----------------
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        face_classes = {}  # stable ID mapping for unknowns
        face_counter = 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            with BytesIO() as buf:
                image.save(buf, format="JPEG")
                faces = recognize_image(buf.getvalue())

            # Assign persistent Person N IDs for unknowns
            for face in faces:
                if face.get("confidence", 0) < 0.6:
                    key = f"{round(face['x']/25)}-{round(face['y']/25)}-{round(face['w']/25)}-{round(face['h']/25)}"
                    if key not in face_classes:
                        face_classes[key] = f"Person {face_counter}"
                        face_counter += 1
                    face["name"] = face_classes[key]

            result_img = draw_boxes(image, faces, is_video=True)
            stframe.image(result_img, use_column_width=True)

        cap.release()
