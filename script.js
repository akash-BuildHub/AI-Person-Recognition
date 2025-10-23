// DOM Elements
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const webcam = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlayCanvas");
const ctx = overlayCanvas.getContext("2d");
const modelStatus = document.getElementById("modelStatus");

// Global variables
let stream = null;
let running = false;
let animationFrameId = null;
let trackedFaces = new Map(); // Using Map for better tracking
let modelsLoaded = false;
let faceCounter = 0;

// ⚡ FASTER SETTINGS - Reduced smoothing and faster detection
const SMOOTHING = 0.2; // Reduced from 0.4 for faster response
const DETECTION_INTERVAL = 80; // Reduced from 150 for more frequent updates
const TRACKING_TIMEOUT = 5; // Faster cleanup of old faces

// ✅ For Render: Use current origin (will work both locally and on Render)
const BACKEND_URL = window.location.origin;

// Alert system
function showAlert(msg) {
  const alertDiv = document.getElementById("alertMessage");
  alertDiv.textContent = msg;
  alertDiv.style.top = "0";
  setTimeout(() => { alertDiv.style.top = "-50px"; }, 3000);
}

// Calculate distance between two points
function distance(p1, p2) {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

// Find the best matching existing face for a new detection
function findBestMatch(newFace, existingFaces) {
  let bestMatch = null;
  let bestDistance = 50; // Maximum distance to consider as same face
  
  const newCenter = {
    x: newFace.x + newFace.w / 2,
    y: newFace.y + newFace.h / 2
  };
  
  for (const [id, existingFace] of existingFaces) {
    const existingCenter = {
      x: existingFace.x + existingFace.w / 2,
      y: existingFace.y + existingFace.h / 2
    };
    
    const dist = distance(newCenter, existingCenter);
    if (dist < bestDistance) {
      bestDistance = dist;
      bestMatch = id;
    }
  }
  
  return bestMatch;
}

// Update tracked faces with new detections
function updateTrackedFaces(newFaces) {
  const currentTime = performance.now();
  
  // Update existing faces or add new ones
  newFaces.forEach(newFace => {
    const bestMatchId = findBestMatch(newFace, trackedFaces);
    
    if (bestMatchId) {
      // Update existing face with minimal smoothing for faster response
      const existingFace = trackedFaces.get(bestMatchId);
      existingFace.x = newFace.x; // Direct assignment for maximum speed
      existingFace.y = newFace.y;
      existingFace.w = newFace.w;
      existingFace.h = newFace.h;
      existingFace.lastSeen = currentTime;
      existingFace.name = newFace.name;
      existingFace.color = newFace.color;
    } else {
      // Add new face
      const faceId = `face-${faceCounter++}`;
      newFace.lastSeen = currentTime;
      trackedFaces.set(faceId, newFace);
    }
  });
  
  // Remove old faces that haven't been seen recently
  const now = performance.now();
  for (const [id, face] of trackedFaces) {
    if (now - face.lastSeen > TRACKING_TIMEOUT * DETECTION_INTERVAL) {
      trackedFaces.delete(id);
    }
  }
  
  return Array.from(trackedFaces.values());
}

// Draw bounding boxes and labels - Optimized for speed
function drawBoxes(faces) {
  if (!running) return;
  
  // Clear canvas
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  
  faces.forEach(face => {
    const isKnown = face.name !== "unknown";
    
    // Draw bounding box
    ctx.strokeStyle = isKnown ? "green" : "red";
    ctx.lineWidth = 3;
    ctx.strokeRect(face.x, face.y, face.w, face.h);
    
    // Draw label
    const label = face.name;
    ctx.font = "bold 16px 'Segoe UI', Arial, sans-serif";
    const textWidth = ctx.measureText(label).width;
    
    // Position label above face, or below if face is at top
    const textX = Math.max(5, Math.min(face.x, overlayCanvas.width - textWidth - 5));
    const textY = face.y > 25 ? face.y - 8 : face.y + face.h + 20;
    
    // Label background
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(textX - 5, textY - 18, textWidth + 10, 20);
    
    // Label text
    ctx.fillStyle = "white";
    ctx.fillText(label, textX, textY - 2);
  });
}

// Main face detection loop - Optimized for speed
let lastDetectionTime = 0;
async function detectFaces() {
  if (!running) return;
  
  const now = Date.now();
  if (now - lastDetectionTime < DETECTION_INTERVAL) {
    animationFrameId = requestAnimationFrame(detectFaces);
    return;
  }
  lastDetectionTime = now;

  try {
    // Capture frame efficiently
    const canvas = document.createElement("canvas");
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    
    // Convert to blob with lower quality for faster transfer
    const blob = await new Promise(resolve => {
      canvas.toBlob(resolve, 'image/jpeg', 0.7); // Reduced quality for speed
    });
    
    const formData = new FormData();
    formData.append('frame', blob, 'frame.jpg');
    
    // Send to backend
    const response = await fetch(`${BACKEND_URL}/detect`, {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.status === 'success') {
      const newFaces = data.faces.map(face => ({
        name: face.name,
        color: face.color,
        x: face.box[0],
        y: face.box[1],
        w: face.box[2],
        h: face.box[3]
      }));
      
      // Update tracking and draw
      const trackedFaces = updateTrackedFaces(newFaces);
      drawBoxes(trackedFaces);
      
      // Update status
      if (trackedFaces.length > 0) {
        const knownFaces = trackedFaces.filter(face => face.name !== "unknown").length;
        const unknownFaces = trackedFaces.filter(face => face.name === "unknown").length;
        modelStatus.innerHTML = `✅ Tracking ${trackedFaces.length} faces (${knownFaces} known, ${unknownFaces} unknown)`;
      } else {
        modelStatus.innerHTML = '🔍 No faces detected';
      }
    } else {
      console.error('Detection error:', data.message);
    }
    
  } catch (error) {
    console.error('❌ Face detection error:', error);
  }

  animationFrameId = requestAnimationFrame(detectFaces);
}

// Webcam management
async function startWebcam() {
  try {
    if (running) return;
    
    if (!modelsLoaded) {
      showAlert('AI backend is not connected. Please wait...');
      return;
    }
    
    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    trackedFaces.clear();
    faceCounter = 0;
    
    // Clear canvas
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    
    stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 }, 
        height: { ideal: 480 },
        frameRate: { ideal: 30 }, // Ensure good frame rate
        facingMode: "user" 
      }, 
      audio: false 
    });
    
    webcam.srcObject = stream;
    
    webcam.onloadedmetadata = () => {
      overlayCanvas.width = webcam.videoWidth;
      overlayCanvas.height = webcam.videoHeight;
      
      console.log('🎥 Webcam started, starting face detection...');
      modelStatus.innerHTML = '🔍 Detecting faces...';
      detectFaces();
    };
    
  } catch (err) {
    console.error('❌ Webcam error:', err);
    showAlert('Cannot access webcam: ' + err.message);
    running = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function stopWebcam() {
  if (!running) return;
  
  running = false;
  cancelAnimationFrame(animationFrameId);
  animationFrameId = null;
  
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  trackedFaces.clear();
  
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  
  webcam.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  modelStatus.innerHTML = '✅ AI Backend connected - Ready for detection';
}

// Initialize backend connection
async function initializeBackend() {
  try {
    modelStatus.innerHTML = '<span class="loading"></span> Connecting to AI backend...';
    
    const response = await fetch(`${BACKEND_URL}/train`, { method: 'GET' });
    if (!response.ok) throw new Error('Backend not responding');
    
    const data = await response.json();
    console.log('✅ Backend connected successfully:', data);
    
    modelsLoaded = true;
    
    if (data.from_cache) {
      modelStatus.innerHTML = `✅ AI Ready! Loaded ${data.face_count} faces for ${data.people_count} people from cache`;
      showAlert(`AI system ready! Recognizes ${data.people_count} people (loaded from cache)`);
    } else {
      modelStatus.innerHTML = `✅ AI Ready! Trained ${data.face_count} faces for ${data.people_count} people`;
      showAlert(`AI system ready! Recognizes ${data.people_count} people`);
    }
    
    startBtn.disabled = false;
    
  } catch (error) {
    console.error('❌ Backend connection error:', error);
    modelStatus.innerHTML = '❌ Failed to connect to backend';
    showAlert('Failed to connect to AI backend. Please check console for details.');
  }
}

// Event listeners
startBtn.addEventListener("click", startWebcam);
stopBtn.addEventListener("click", stopWebcam);

// Initialize the application
window.addEventListener('load', async () => {
  console.log('🚀 Page loaded, initializing face recognition...');
  console.log('📍 Backend URL:', BACKEND_URL);
  
  startBtn.disabled = true;
  stopBtn.disabled = true;
  
  await initializeBackend();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  stopWebcam();
});