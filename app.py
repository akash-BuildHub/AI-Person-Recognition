from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
KNOWN_FACES_DIR = "data"
TOLERANCE = 0.6
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Global variables
known_face_encodings = []
known_face_names = []

def train_face_recognition():
    global known_face_encodings, known_face_names
    
    known_face_encodings = []
    known_face_names = []
    
    logger.info("🔍 Training face recognition model...")
    
    try:
        import face_recognition
    except ImportError:
        logger.error("❌ face_recognition library not installed.")
        return False
    
    if not os.path.exists(KNOWN_FACES_DIR):
        logger.warning(f"📁 Directory {KNOWN_FACES_DIR} not found")
        return True
    
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        logger.info(f"👤 Training on {person_name}...")
        
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_dir, filename)
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                        logger.info(f"✅ Learned face from {filename}")
                    else:
                        logger.warning(f"⚠️ No face found in {filename}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {filename}: {str(e)}")
    
    logger.info(f"✅ Training complete! Learned {len(known_face_names)} faces for {len(set(known_face_names))} people")
    return True

def recognize_faces(frame):
    try:
        import face_recognition
    except ImportError:
        return []
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    faces_data = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "unknown"
        color = "red"
        
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
            
            if best_match_index != -1 and matches[best_match_index]:
                name = known_face_names[best_match_index]
                color = "green"
        
        face_data = {
            "name": name,
            "color": color,
            "box": [left, top, right - left, bottom - top],
            "confidence": float(1 - face_distances[best_match_index]) if best_match_index != -1 and name != "unknown" else 0.0
        }
        
        faces_data.append(face_data)
    
    return faces_data

# Serve the main HTML page
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Serve static files (CSS, JS)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train or retrain the face recognition model"""
    try:
        success = train_face_recognition()
        if success:
            return jsonify({
                "status": "success",
                "message": f"Model trained with {len(known_face_names)} faces",
                "face_count": len(known_face_encodings),
                "people_count": len(set(known_face_names))
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to train model. Check if face_recognition library is installed."
            }), 500
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Training failed: {str(e)}"
        }), 500

@app.route('/detect', methods=['POST'])
def detect_faces():
    """Detect and recognize faces in an uploaded frame"""
    try:
        if 'frame' not in request.files:
            return jsonify({"error": "No frame provided"}), 400
        
        file = request.files['frame']
        img_array = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        faces_data = recognize_faces(frame)
        
        return jsonify({
            "status": "success",
            "faces": faces_data,
            "face_count": len(faces_data)
        })
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Detection failed: {str(e)}"
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "message": "AI Face Recognition API is running"})

if __name__ == '__main__':
    train_face_recognition()
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting AI Live Face Recognition Server...")
    print(f"📁 Data directory: {KNOWN_FACES_DIR}")
    print(f"🔗 Server running on port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)