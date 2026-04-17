# Dental Disease Detection – README
# ===================================

## Project Structure

dental-detection/
├── app.py                  # Flask application entry point
├── requirements.txt        # Python dependencies
├── models/
│   └── best.pt             ← PLACE YOUR TRAINED DEEP LEARNING MODEL HERE
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/            # Auto-created; stores uploaded & annotated images
├── templates/
│   ├── index.html          # Main detection page
│   └── about.html          # About page
└── utils/
    └── detector.py         # deep learning detector wrapper

## Setup & Run

### 1. Place your model
Copy your trained deep learning weights file into the models/ folder and name it:
    models/best.pt

### 2. Create a virtual environment (recommended)
    python -m venv venv
    venv\Scripts\activate          # Windows
    source venv/bin/activate       # Linux / macOS

### 3. Install dependencies
    pip install -r requirements.txt

### 4. Run the app
    python app.py

### 5. Open in browser
    http://localhost:5000

## Usage
1. Go to http://localhost:5000
2. Upload a dental X-ray or intra-oral photo (PNG / JPG / BMP / WEBP, max 16 MB)
3. Click "Run Detection"
4. View the annotated image with bounding boxes + detailed findings

## Supported Disease Classes (from your model)
- Caries (Tooth Decay)
- Calculus (Tartar)
- Gingivitis
- Ulcers
- Hypodontia
(Add or modify classes in utils/detector.py → DISEASE_INFO)

## Notes
- The app automatically loads the deep learning backend via torch.hub on first run.
- If  models/best.pt  is missing, the app still starts but shows a placeholder message.
- Uploaded images are saved in static/uploads/ — clear this folder periodically.

## Disclaimer
For educational and research purposes only.
Not a substitute for professional clinical diagnosis.
