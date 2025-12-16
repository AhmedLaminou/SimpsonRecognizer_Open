// ===================================
// SIMPSON KNOWER - Character Classifier
// Springfield's Smartest AI!
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
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const navLinks = document.querySelector('.nav-links');

// State
let currentFile = null;

// ===== MOBILE MENU =====
if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        navLinks.classList.toggle('active');
    });
}

// ===== CHARACTER EMOJIS =====
const characterEmojis = {
    'homer_simpson': '👨‍🦲',
    'bart_simpson': '👦',
    'marge_simpson': '👩',
    'lisa_simpson': '👧',
    'maggie_simpson': '👶',
    'ned_flanders': '🧔',
    'charles_montgomery_burns': '👴',
    'moe_szyslak': '🍺',
    'krusty_the_clown': '🤡',
    'principal_skinner': '👨‍🏫'
};

// ===== FILE UPLOAD HANDLERS =====
// Only attach events if we're on the main page with upload area
if (uploadArea && fileInput) {
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
}

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
if (removeBtn) {
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });
}

function resetUpload() {
    if (fileInput) fileInput.value = '';
    currentFile = null;
    if (imagePreview) imagePreview.src = '';
    if (previewContainer) previewContainer.classList.remove('active');
    if (predictBtn) predictBtn.classList.remove('active');
    hideResults();
    hideError();
}

// ===== PREDICTION =====
if (predictBtn) {
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
}

// ===== DISPLAY RESULTS =====
function displayResults(data) {
    // Character name
    const characterName = document.getElementById('characterName');
    characterName.textContent = data.character.replace(/_/g, ' ');
    
    // Character emoji icon
    const characterIcon = document.getElementById('characterIcon');
    if (characterIcon) {
        characterIcon.textContent = characterEmojis[data.character] || '👤';
    }
    
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
        const emoji = characterEmojis[char] || '👤';
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="char-name">${emoji} ${char.replace(/_/g, ' ')}</span>
            <span class="char-confidence">${conf}</span>
        `;
        predictionsList.appendChild(item);
    });
    
    // Show results
    showResults();
}

// ===== UI STATE MANAGEMENT =====
function showLoading() {
    if (loadingContainer) loadingContainer.classList.add('active');
}

function hideLoading() {
    if (loadingContainer) loadingContainer.classList.remove('active');
}

function showResults() {
    if (resultsSection) {
        resultsSection.classList.add('active');
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
}

function hideResults() {
    if (resultsSection) resultsSection.classList.remove('active');
}

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    if (errorMessage) errorMessage.textContent = message;
    if (errorContainer) {
        errorContainer.classList.add('active');
        // Scroll to error
        setTimeout(() => {
            errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
}

function hideError() {
    if (errorContainer) errorContainer.classList.remove('active');
}

// ===== ACTION BUTTONS =====
if (tryAgainBtn) {
    tryAgainBtn.addEventListener('click', () => {
        resetUpload();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

if (dismissErrorBtn) {
    dismissErrorBtn.addEventListener('click', () => {
        hideError();
    });
}

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', (e) => {
    // Escape key to reset
    if (e.key === 'Escape') {
        if (resultsSection && resultsSection.classList.contains('active')) {
            resetUpload();
        } else if (errorContainer && errorContainer.classList.contains('active')) {
            hideError();
        }
    }
    
    // Enter key to predict (if image is loaded)
    if (e.key === 'Enter' && currentFile && predictBtn && !predictBtn.disabled) {
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
console.log('🍩 Simpson Knower initialized!');
console.log('🎭 Springfield\'s smartest AI is ready!');
console.log('📸 Upload an image to identify a character');
console.log('D\'oh! - Homer Simpson');
