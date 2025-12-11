"""
Simpsons Character Classifier - Flask Backend
A clean Flask API for classifying Simpsons characters from uploaded images
"""

from flask import Flask, render_template, request, jsonify
import os
import cv2 as cv
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static', 
            static_url_path='/static',
            template_folder='static')

# Load the trained model
print("🔄 Loading model...")
try:
    model = load_model('simpsons_classifier.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not load model - {e}")
    print("    Make sure 'simpsons_classifier.h5' exists in the project directory")
    model = None

# Configuration
IMG_SIZE = (80, 80)
CHANNELS = 1  # Grayscale

# Character classes (must match training order)
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

# ===== IMAGE PREPROCESSING =====
def prepare_image(image):
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array ready for model prediction
    """
    try:
        # Convert PIL Image to numpy array
        image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        
        # Resize to model's expected input size
        image = cv.resize(image, IMG_SIZE)
        
        # Normalize pixel values to 0-1
        image = image.astype('float32') / 255.0
        
        # Reshape to match model input: (1, height, width, channels)
        image = image.reshape(1, IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
        
        return image
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# ===== PREDICTION =====
def predict_character(image):
    """
    Make prediction on uploaded image
    
    Args:
        image: PIL Image object
        
    Returns:
        dict with prediction results
    """
    if model is None:
        return {
            'error': 'Model not loaded. Please ensure simpsons_classifier.h5 exists.'
        }
    
    try:
        # Preprocess the image
        processed_image = prepare_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Format results
        return {
            'character': characters[predicted_class],
            'confidence': f"{confidence * 100:.2f}%",
            'all_predictions': {
                char: f"{float(predictions[0][i]) * 100:.2f}%" 
                for i, char in enumerate(characters)
            }
        }
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}

# ===== ROUTES =====
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and prediction
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with prediction results
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF'}), 400
        
        # Read and process the image
        image = Image.open(file.stream)
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Make prediction
        result = predict_character(image)
        
        # Check for errors
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': len(characters)
    })

# ===== MAIN =====
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎭 SIMPSONS CHARACTER CLASSIFIER")
    print("="*60)
    print(f"📍 Server: http://localhost:5000")
    print(f"🧠 Model: {'Loaded ✅' if model else 'Not loaded ⚠️'}")
    print(f"🎯 Classes: {len(characters)}")
    print("="*60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)