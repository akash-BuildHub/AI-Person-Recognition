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
let trackedFaces = [];
let modelsLoaded = false;

const SMOOTHING = 0.4;
const DETECTION_INTERVAL = 150;

// ✅ FIXED: Use the same origin since Flask now serves the HTML
const BACKEND_URL = window.location.origin;

// Alert system
function showAlert(msg) {
  const alertDiv = document.getElementById("alertMessage");
  alertDiv.textContent = msg;
  alertDiv.style.top = "0";
  setTimeout(() => { alertDiv.style.top = "-50px"; }, 3000);
}

// Draw bounding boxes and labels
function drawBoxes(faces) {
  if (!running) return;
  
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  
  faces.forEach(face => {
    // Smooth the box coordinates
    const prev = trackedFaces.find(f => f.name === face.name && Math.abs(f.x - face.x) < 50) || {};
    const x = prev.x !== undefined ? prev.x + SMOOTHING * (face.x - prev.x) : face.x;
    const y = prev.y !== undefined ? prev.y + SMOOTHING * (face.y - prev.y) : face.y;
    const w = prev.w !== undefined ? prev.w + SMOOTHING * (face.w - prev.w) : face.w;
    const h = prev.h !== undefined ? prev.h + SMOOTHING * (face.h - prev.h) : face.h;

    // Set colors: Green for known, Red for unknown
    const isKnown = face.name !== "unknown";
    ctx.strokeStyle = isKnown ? "green" : "red";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
    
    // Draw label background
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "bold 18px 'Segoe UI'";
    const textWidth = ctx.measureText(face.name).width;
    const textX = x > 10 ? x : 10;
    const textY = y > 25 ? y - 8 : y + 25;
    
    // Draw label with background
    ctx.fillRect(textX - 5, textY - 20, textWidth + 10, 25);
    ctx.fillStyle = "white";
    ctx.fillText(face.name, textX, textY);

    // Update tracked face with smoothed coordinates
    face.x = x; face.y = y; face.w = w; face.h = h;
  });
  
  trackedFaces = faces;
}

// Main face detection loop
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
    const frame = captureFrame();
    const blob = await new Promise(resolve => {
      frame.toBlob(resolve, 'image/jpeg', 0.8);
    });
    
    const formData = new FormData();
    formData.append('frame', blob, 'frame.jpg');
    
    const response = await fetch(`${BACKEND_URL}/detect`, {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.status === 'success') {
      const faces = data.faces.map(face => ({
        name: face.name,
        x: face.box[0],
        y: face.box[1],
        w: face.box[2],
        h: face.box[3]
      }));
      
      drawBoxes(faces);
    } else {
      console.error('Detection error:', data.message);
    }
    
  } catch (error) {
    console.error('❌ Face detection error:', error);
  }

  animationFrameId = requestAnimationFrame(detectFaces);
}

function captureFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = webcam.videoWidth;
  canvas.height = webcam.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
  return canvas;
}

// Webcam management
async function startWebcam() {
  try {
    if (running) return;
    
    // Ensure backend is connected before starting webcam
    if (!modelsLoaded) {
      showAlert('AI backend is not connected. Please wait...');
      return;
    }
    
    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    trackedFaces = [];
    
    stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 }, 
        height: { ideal: 480 },
        facingMode: "user" 
      }, 
      audio: false 
    });
    
    webcam.srcObject = stream;
    
    webcam.onloadedmetadata = () => {
      // Set canvas size to match video
      overlayCanvas.width = webcam.videoWidth;
      overlayCanvas.height = webcam.videoHeight;
      
      console.log('🎥 Webcam started, starting face detection...');
      // Start face detection
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
  trackedFaces = [];
  
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  
  webcam.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

// Initialize backend connection
async function initializeBackend() {
  try {
    modelStatus.innerHTML = '<span class="loading"></span> Connecting to AI backend...';
    
    // Test backend connection
    const response = await fetch(`${BACKEND_URL}/train`, { method: 'GET' });
    if (!response.ok) throw new Error('Backend not responding');
    
    console.log('✅ Backend connected successfully');
    modelsLoaded = true;
    modelStatus.innerHTML = '✅ AI Backend connected successfully!';
    showAlert('AI backend connected successfully!');
    
    // Enable start button now that backend is connected
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
  
  // Start with Start button disabled
  startBtn.disabled = true;
  stopBtn.disabled = true;
  
  await initializeBackend();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  stopWebcam();
});