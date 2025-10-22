const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');
const faceapi = require('face-api.js');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const PORT = process.env.PORT || 3000;

// Global variables for face recognition
let faceMatcher = null;
let isModelsLoaded = false;

// Load face-api.js models
async function loadModels() {
  try {
    console.log('Loading face detection models...');
    
    const modelPath = './models';
    await faceapi.nets.tinyFaceDetector.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    
    console.log('All models loaded successfully');
    isModelsLoaded = true;
    
    // Train face recognizer with known faces
    await trainFaceRecognizer();
  } catch (error) {
    console.error('Error loading models:', error);
  }
}

// Train face recognizer with known faces
async function trainFaceRecognizer() {
  try {
    console.log('Training face recognizer...');
    const labels = ['Akash', 'Maria']; // Add your known person labels
    
    const labeledFaceDescriptors = await Promise.all(
      labels.map(async label => {
        // Load reference images for each person
        const imgUrl = `./data/${label}/0.jpg`;
        const img = await canvas.loadImage(imgUrl);
        
        // Detect face and compute descriptor
        const detection = await faceapi
          .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptor();
        
        if (!detection) {
          throw new Error(`No face detected in reference image for ${label}`);
        }
        
        return new faceapi.LabeledFaceDescriptors(label, [detection.descriptor]);
      })
    );
    
    faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
    console.log('Face recognizer trained successfully');
  } catch (error) {
    console.error('Error training face recognizer:', error);
  }
}

// Face recognition endpoint
app.post('/recognize', upload.single('image'), async (req, res) => {
  try {
    if (!isModelsLoaded || !faceMatcher) {
      return res.status(503).json({ 
        error: 'Models not loaded yet. Please try again in a moment.' 
      });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    // Convert buffer to image
    const img = new Image();
    img.src = req.file.buffer;

    // Detect all faces in the image
    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    const faces = detections.map(detection => {
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      
      // Get bounding box coordinates
      const box = detection.detection.box;
      
      return {
        name: bestMatch.label,
        confidence: bestMatch.distance,
        x: box.x,
        y: box.y,
        w: box.width,
        h: box.height
      };
    });

    res.json({ faces });
  } catch (error) {
    console.error('Recognition error:', error);
    res.status(500).json({ error: 'Face recognition failed' });
  }
});

// Serve static files
app.use(express.static('.'));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    modelsLoaded: isModelsLoaded,
    faceMatcherReady: !!faceMatcher 
  });
});

// Start server
app.listen(PORT, async () => {
  console.log(`Server running on port ${PORT}`);
  console.log('Loading AI models... This may take a few seconds.');
  await loadModels();
});