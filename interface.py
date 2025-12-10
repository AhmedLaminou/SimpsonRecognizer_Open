<<<<<<< HEAD
"""
Simpsons Character Classifier - Web Interface
A Flask web app for classifying Simpsons characters from uploaded images
"""

from flask import Flask, render_template, request, jsonify
import os
import cv2 as cv
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
model = load_model('simpsons_classifier.h5')

app = Flask(__name__, 
            static_folder='static', 
            static_url_path='/static',
            template_folder='static')

# Configuration
IMG_SIZE = (80, 80)
characters = [
    'homer_simpson',
    'bart_simpson', 
    'marge_simpson',
    'lisa_simpson',
    'maggie_simpson',
    'ned_flanders',
    'charles_montgomery_burns',
    'moe_szyslak',
    'krusty_the_clown',
    'principal_skinner'
] 

def prepare_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    # Resize to model's expected input size
    image = cv.resize(image, IMG_SIZE)
    
    # Normalize pixel values to 0-1
    image = image / 255.0
    
    # Reshape to match model input: (1, height, width, channels)
    image = image.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    
    return image

def predict_character(image):
    """Make prediction on uploaded image"""
    try:
        # Preprocess the image
        processed_image = prepare_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return {
            'character': characters[predicted_class],
            'confidence': f"{confidence * 100:.2f}%",
            'all_predictions': {
                char: f"{float(predictions[0][i]):.2%}" 
                for i, char in enumerate(characters)
            }
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process the image
        image = Image.open(file.stream)
        
        # Make prediction
        result = predict_character(image)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎭 Starting Simpsons Character Classifier...")
    print("📍 Open http://localhost:5000 in your browser")
    print("    Model loaded from: simpsons_classifier.h5")
=======
"""
Simpsons Character Classifier - Web Interface
A Flask web app for classifying Simpsons characters from uploaded images
"""

from flask import Flask, render_template_string, request, jsonify
import os
import cv2 as cv
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# NOTE: You'll need to load your trained model here
# Uncomment and modify this line with your actual model path:
# from tensorflow.keras.models import load_model
# model = load_model('path_to_your_saved_model.h5')

app = Flask(__name__)

# Configuration
IMG_SIZE = (80, 80)
characters = [
    'homer_simpson',
    'bart_simpson', 
    'marge_simpson',
    'lisa_simpson',
    'maggie_simpson',
    'ned_flanders',
    'charles_montgomery_burns',
    'moe_szyslak',
    'krusty_the_clown',
    'principal_skinner'
]  # Replace with your actual character list

def prepare_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    # Resize to model's expected input size
    image = cv.resize(image, IMG_SIZE)
    
    # Normalize pixel values to 0-1
    image = image / 255.0
    
    # Reshape to match model input: (1, height, width, channels)
    image = image.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    
    return image

def predict_character(image):
    """Make prediction on uploaded image"""
    try:
        # Preprocess the image
        processed_image = prepare_image(image)
        
        # Make prediction (UNCOMMENT when you have your model loaded)
        # predictions = model.predict(processed_image)
        # predicted_class = np.argmax(predictions[0])
        # confidence = float(predictions[0][predicted_class])
        
        # DEMO MODE (remove this when you load your real model)
        # Simulating a prediction
        predicted_class = np.random.randint(0, len(characters))
        confidence = np.random.uniform(0.7, 0.99)
        
        return {
            'character': characters[predicted_class],
            'confidence': f"{confidence * 100:.2f}%",
            'all_predictions': {
                char: f"{np.random.uniform(0, confidence):.2%}" 
                for char in characters
            }
        }
    except Exception as e:
        return {'error': str(e)}

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simpsons Character Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }

        h1 {
            text-align: center;
            color: #1e88e5;
            margin-bottom: 10px;
            font-size: 2.5em;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }

        .upload-area {
            border: 3px dashed #1e88e5;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }

        .upload-area:hover {
            background: #e3f2fd;
            border-color: #0d47a1;
        }

        .upload-area.dragover {
            background: #bbdefb;
            border-color: #0d47a1;
        }

        #fileInput {
            display: none;
        }

        .upload-icon {
            font-size: 4em;
            margin-bottom: 15px;
        }

        .upload-text {
            font-size: 1.2em;
            color: #555;
            margin-bottom: 10px;
        }

        .upload-hint {
            font-size: 0.9em;
            color: #999;
        }

        #imagePreview {
            max-width: 100%;
            max-height: 400px;
            margin: 20px auto;
            display: none;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        #predictBtn {
            background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 30px;
            cursor: pointer;
            display: none;
            margin: 20px auto;
            transition: transform 0.2s, box-shadow 0.2s;
            font-weight: bold;
        }

        #predictBtn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(30, 136, 229, 0.3);
        }

        #predictBtn:active {
            transform: translateY(0);
        }

        #predictBtn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        #results {
            display: none;
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 5px solid #1e88e5;
        }

        .result-character {
            font-size: 2em;
            color: #1e88e5;
            font-weight: bold;
            margin-bottom: 10px;
            text-transform: capitalize;
        }

        .result-confidence {
            font-size: 1.3em;
            color: #4caf50;
            margin-bottom: 20px;
        }

        .all-predictions {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
        }

        .prediction-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 8px;
            transition: background 0.2s;
        }

        .prediction-item:hover {
            background: #e3f2fd;
        }

        .char-name {
            font-weight: 500;
            text-transform: capitalize;
        }

        .char-confidence {
            color: #666;
            font-weight: bold;
        }

        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #1e88e5;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }

        .btn-container {
            text-align: center;
        }

        #resetBtn {
            background: #f44336;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 1em;
            border-radius: 25px;
            cursor: pointer;
            margin-top: 15px;
            transition: background 0.3s;
        }

        #resetBtn:hover {
            background: #d32f2f;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Simpsons Character Classifier</h1>
        <p class="subtitle">Upload an image and let AI identify the character!</p>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Click to upload or drag and drop</div>
            <div class="upload-hint">Supports: JPG, PNG, GIF</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>

        <img id="imagePreview" alt="Preview">

        <div class="btn-container">
            <button id="predictBtn">🔍 Identify Character</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image...</p>
        </div>

        <div class="error" id="error"></div>

        <div id="results">
            <div class="result-character" id="resultCharacter"></div>
            <div class="result-confidence" id="resultConfidence"></div>
            
            <div class="all-predictions">
                <h3 style="margin-bottom: 15px; color: #555;">All Predictions:</h3>
                <div id="allPredictions"></div>
            </div>

            <div class="btn-container">
                <button id="resetBtn">↻ Try Another Image</button>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const imagePreview = document.getElementById('imagePreview');
        const predictBtn = document.getElementById('predictBtn');
        const results = document.getElementById('results');
        const loading = document.getElementById('loading');
        const errorDiv = document.getElementById('error');
        const resetBtn = document.getElementById('resetBtn');

        let currentFile = null;

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', handleFileSelect);

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) {
                handleFile(file);
            }
        }

        function handleFile(file) {
            currentFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                predictBtn.style.display = 'block';
                results.style.display = 'none';
                errorDiv.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        // Predict button
        predictBtn.addEventListener('click', async () => {
            if (!currentFile) return;

            // Show loading
            loading.style.display = 'block';
            predictBtn.disabled = true;
            results.style.display = 'none';
            errorDiv.style.display = 'none';

            // Prepare form data
            const formData = new FormData();
            formData.append('image', currentFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                // Display results
                document.getElementById('resultCharacter').textContent = 
                    data.character.replace(/_/g, ' ');
                document.getElementById('resultConfidence').textContent = 
                    `Confidence: ${data.confidence}`;

                // Display all predictions
                const allPredDiv = document.getElementById('allPredictions');
                allPredDiv.innerHTML = '';
                
                for (const [char, conf] of Object.entries(data.all_predictions)) {
                    const item = document.createElement('div');
                    item.className = 'prediction-item';
                    item.innerHTML = `
                        <span class="char-name">${char.replace(/_/g, ' ')}</span>
                        <span class="char-confidence">${conf}</span>
                    `;
                    allPredDiv.appendChild(item);
                }

                results.style.display = 'block';
            } catch (error) {
                errorDiv.textContent = `Error: ${error.message}`;
                errorDiv.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                predictBtn.disabled = false;
            }
        });

        // Reset button
        resetBtn.addEventListener('click', () => {
            fileInput.value = '';
            imagePreview.style.display = 'none';
            predictBtn.style.display = 'none';
            results.style.display = 'none';
            errorDiv.style.display = 'none';
            currentFile = null;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process the image
        image = Image.open(file.stream)
        
        # Make prediction
        result = predict_character(image)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎭 Starting Simpsons Character Classifier...")
    print("📍 Open http://localhost:5000 in your browser")
    print("\n⚠️  NOTE: This is running in DEMO mode!")
    print("    To use your trained model:")
    print("    1. Uncomment the model loading lines at the top")
    print("    2. Save your trained model: model.save('simpsons_model.h5')")
    print("    3. Update the model path in the code")
>>>>>>> c0f8356393cf0dd42b306eb133e2f07070df0306
    app.run(debug=True, port=5000)