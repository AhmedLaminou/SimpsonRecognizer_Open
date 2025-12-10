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
    app.run(debug=True, port=5000)