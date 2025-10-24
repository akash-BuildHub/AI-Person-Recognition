from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import logging
import pickle
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
KNOWN_FACES_DIR = "data"
ENCODINGS_FILE = "face_encodings.pkl"
TOLERANCE = 0.6
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Global variables
known_face_encodings = []
known_face_names = []
model_loaded = False

def train_face_recognition(force_retrain=False):
    global known_face_encodings, known_face_names, model_loaded
    
    # Try to load from cache first
    if not force_retrain and os.path.exists(ENCODINGS_FILE):
        try:
            logger.info("🔍 Loading cached face encodings...")
            with open(ENCODINGS_FILE, "rb") as f:
                cache_data = pickle.load(f)
                known_face_encodings = cache_data['encodings']
                known_face_names = cache_data['names']
            
            logger.info(f"✅ Loaded {len(known_face_encodings)} face encodings from cache")
            model_loaded = True
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache: {str(e)}")
    
    # If cache doesn't exist or loading failed
    known_face_encodings = []
    known_face_names = []
    model_loaded = False
    
    logger.info("🔍 Training face recognition model...")
    
    try:
        import face_recognition
    except ImportError as e:
        logger.error(f"❌ face_recognition library not installed: {str(e)}")
        return False
    
    if not os.path.exists(KNOWN_FACES_DIR):
        logger.warning(f"📁 Directory {KNOWN_FACES_DIR} not found")
        # Return True to allow the app to run without training data
        return True
    
    trained_count = 0
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
                    
                    for encoding in face_encodings:
                        known_face_encodings.append(encoding)
                        known_face_names.append(person_name)
                        trained_count += 1
                    
                    if face_encodings:
                        logger.info(f"✅ Learned {len(face_encodings)} face(s) from {filename}")
                    else:
                        logger.warning(f"⚠️ No face found in {filename}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {filename}: {str(e)}")
    
    # Save to cache for next time
    if known_face_encodings:
        try:
            cache_data = {
                'encodings': known_face_encodings,
                'names': known_face_names
            }
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"💾 Saved {len(known_face_encodings)} encodings to cache")
        except Exception as e:
            logger.warning(f"⚠️ Could not save cache: {str(e)}")
    
    model_loaded = len(known_face_encodings) > 0
    unique_people = len(set(known_face_names))
    logger.info(f"✅ Training complete! Learned {len(known_face_encodings)} faces for {unique_people} people")
    return True

def recognize_faces(frame):
    if not model_loaded:
        return []
        
    try:
        import face_recognition
    except ImportError:
        return []
    
    try:
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
                "box": [left, top, right - left, bottom - top]
            }
            
            faces_data.append(face_data)
        
        return faces_data
    except Exception as e:
        logger.error(f"❌ Face recognition error: {str(e)}")
        return []

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
        force_retrain = request.method == 'POST'
        success = train_face_recognition(force_retrain=force_retrain)
        
        if success:
            people_count = len(set(known_face_names)) if known_face_names else 0
            message = "Model loaded from cache" if not force_retrain and os.path.exists(ENCODINGS_FILE) else "Model trained successfully"
            
            return jsonify({
                "status": "success",
                "message": f"{message} with {len(known_face_encodings)} face encodings",
                "face_count": len(known_face_encodings),
                "people_count": people_count,
                "people": list(set(known_face_names)) if known_face_names else [],
                "from_cache": not force_retrain and os.path.exists(ENCODINGS_FILE)
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

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the face encodings cache"""
    try:
        if os.path.exists(ENCODINGS_FILE):
            os.remove(ENCODINGS_FILE)
            logger.info("🗑️ Cleared face encodings cache")
            return jsonify({"status": "success", "message": "Cache cleared successfully"})
        else:
            return jsonify({"status": "success", "message": "No cache file found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get current model statistics"""
    cache_exists = os.path.exists(ENCODINGS_FILE)
    return jsonify({
        "status": "success",
        "known_people": len(set(known_face_names)) if known_face_names else 0,
        "total_encodings": len(known_face_encodings),
        "people_list": list(set(known_face_names)) if known_face_names else [],
        "cache_loaded": cache_exists,
        "cache_file": ENCODINGS_FILE if cache_exists else None,
        "model_loaded": model_loaded
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "AI Face Recognition API is running",
        "model_loaded": model_loaded,
        "cache_used": os.path.exists(ENCODINGS_FILE),
        "python_version": sys.version.split()[0]
    })

# Initialize the model when the app starts
try:
    logger.info("🚀 Initializing face recognition model...")
    train_face_recognition(force_retrain=False)
except Exception as e:
    logger.error(f"❌ Failed to initialize model: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting AI Live Face Recognition Server...")
    print(f"📁 Data directory: {KNOWN_FACES_DIR}")
    print(f"💾 Cache file: {ENCODINGS_FILE}")
    print(f"👤 Known people: {len(set(known_face_names)) if known_face_names else 0}")
    print(f"🔢 Total encodings: {len(known_face_encodings)}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"🔗 Server running on port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)