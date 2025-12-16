// ===================================
// SIMPSONS CHARACTER CLASSIFIER
// Interactive JavaScript Module
// ===================================

// ===== DOM ELEMENTS =====
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('previewContainer');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const loadingContainer = document.getElementById('loadingContainer');
const resultsSection = document.getElementById('resultsSection');
const errorContainer = document.getElementById('errorContainer');
const tryAgainBtn = document.getElementById('tryAgainBtn');
const dismissErrorBtn = document.getElementById('dismissErrorBtn');

// State
let currentFile = null;

// ===== PARTICLE ANIMATION =====
class ParticleSystem {
    constructor() {
        this.canvas = document.getElementById('particleCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.particleCount = 80;
        this.connectionDistance = 150;
        
        this.resize();
        this.init();
        this.animate();
        
        window.addEventListener('resize', () => this.resize());
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    init() {
        this.particles = [];
        for (let i = 0; i < this.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 1
            });
        }
    }
    
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Update and draw particles
        this.particles.forEach((particle, i) => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Bounce off edges
            if (particle.x < 0 || particle.x > this.canvas.width) particle.vx *= -1;
            if (particle.y < 0 || particle.y > this.canvas.height) particle.vy *= -1;
            
            // Draw particle
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = 'rgba(0, 212, 255, 0.6)';
            this.ctx.fill();
            
            // Draw connections
            for (let j = i + 1; j < this.particles.length; j++) {
                const other = this.particles[j];
                const dx = particle.x - other.x;
                const dy = particle.y - other.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < this.connectionDistance) {
                    const opacity = (1 - distance / this.connectionDistance) * 0.3;
                    this.ctx.beginPath();
                    this.ctx.strokeStyle = `rgba(0, 212, 255, ${opacity})`;
                    this.ctx.lineWidth = 1;
                    this.ctx.moveTo(particle.x, particle.y);
                    this.ctx.lineTo(other.x, other.y);
                    this.ctx.stroke();
                }
            }
        });
        
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize particle system
const particleSystem = new ParticleSystem();

// ===== FILE UPLOAD HANDLERS =====
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

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
    } else {
        showError('Please upload a valid image file (JPG, PNG, GIF)');
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// ===== FILE HANDLING =====
function handleFile(file) {
    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.classList.add('active');
        predictBtn.classList.add('active');
        hideResults();
        hideError();
    };
    reader.readAsDataURL(file);
}

// Remove image
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

function resetUpload() {
    fileInput.value = '';
    currentFile = null;
    imagePreview.src = '';
    previewContainer.classList.remove('active');
    predictBtn.classList.remove('active');
    hideResults();
    hideError();
}

// ===== PREDICTION =====
predictBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    // Show loading
    showLoading();
    hideResults();
    hideError();
    predictBtn.disabled = true;
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', currentFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to analyze image. Please try again.');
    } finally {
        hideLoading();
        predictBtn.disabled = false;
    }
});

// ===== DISPLAY RESULTS =====
function displayResults(data) {
    // Character name
    const characterName = document.getElementById('characterName');
    characterName.textContent = data.character.replace(/_/g, ' ');
    
    // Confidence
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    
    // Parse confidence percentage
    const confidencePercent = parseFloat(data.confidence);
    confidenceValue.textContent = data.confidence;
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = `${confidencePercent}%`;
    }, 100);
    
    // All predictions
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';
    
    // Sort predictions by confidence
    const sortedPredictions = Object.entries(data.all_predictions)
        .sort((a, b) => {
            const aVal = parseFloat(a[1]);
            const bVal = parseFloat(b[1]);
            return bVal - aVal;
        });
    
    sortedPredictions.forEach(([char, conf]) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="char-name">${char.replace(/_/g, ' ')}</span>
            <span class="char-confidence">${conf}</span>
        `;
        predictionsList.appendChild(item);
    });
    
    // Show results
    showResults();
}

// ===== UI STATE MANAGEMENT =====
function showLoading() {
    loadingContainer.classList.add('active');
}

function hideLoading() {
    loadingContainer.classList.remove('active');
}

function showResults() {
    resultsSection.classList.add('active');
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideResults() {
    resultsSection.classList.remove('active');
}

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorContainer.classList.add('active');
    
    // Scroll to error
    setTimeout(() => {
        errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideError() {
    errorContainer.classList.remove('active');
}

// ===== ACTION BUTTONS =====
tryAgainBtn.addEventListener('click', () => {
    resetUpload();
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

dismissErrorBtn.addEventListener('click', () => {
    hideError();
});

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', (e) => {
    // Escape key to reset
    if (e.key === 'Escape') {
        if (resultsSection.classList.contains('active')) {
            resetUpload();
        } else if (errorContainer.classList.contains('active')) {
            hideError();
        }
    }
    
    // Enter key to predict (if image is loaded)
    if (e.key === 'Enter' && currentFile && !predictBtn.disabled) {
        predictBtn.click();
    }
});

// ===== UTILITY FUNCTIONS =====
function formatCharacterName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// ===== INITIALIZATION =====
console.log('🎭 Simpsons Character Classifier initialized');
console.log('✨ Particle system active');
console.log('📸 Ready to analyze images');
