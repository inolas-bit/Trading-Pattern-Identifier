
# app.py — Stable & Fully Synced Version

from flask import Flask, render_template, request
import os, sys, traceback
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image


# Flask App Setup

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/pattern_model.h5'
LABELS_PATH = 'model/labels.txt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Load Model Safely

try:
    model = load_model(MODEL_PATH, compile=False)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f" Error loading model '{MODEL_PATH}' — {e}", file=sys.stderr)
    model = None



# Load Labels from labels.txt

if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, 'r') as f:
            CLASS_LABELS = {
                int(line.split(':')[0]): line.split(':')[1].strip()
                for line in f if ':' in line
            }
        print(" Loaded labels:", CLASS_LABELS)
    except Exception as e:
        print(" Could not parse labels.txt properly:", e)
        CLASS_LABELS = {0: "double_top", 1: "head_shoulders", 2: "no_pattern"}
else:
    print("labels.txt not found — using default mapping")
    CLASS_LABELS = {0: "double_top", 1: "head_shoulders", 2: "no_pattern"}



# Helper: Image Preprocessing

def safe_load_image(img_path, target_size=(224, 224)):
    """Safely loads and preprocesses an image for prediction"""
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.asarray(img).astype('float32') / 255.0
        return arr
    except Exception as e:
        print(f" Error loading image {img_path}: {e}", file=sys.stderr)
        return None


# Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', pattern=" Model not loaded. Please check console.")

    if 'file' not in request.files:
        return render_template('index.html', pattern=" No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', pattern=" No file selected.")

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess & Predict
    try:
        x = safe_load_image(filepath)
        if x is None:
            return render_template('index.html', pattern="❌ Could not process image.", filename=None)

        x = np.expand_dims(x, axis=0)  # shape -> (1, 224, 224, 3)
        preds = model.predict(x)
        print("DEBUG: Prediction array:", preds)

        # Handle unexpected model output
        if preds.ndim != 2:
            return render_template(
                'index.html',
                pattern="❌ Invalid model output shape.",
                filename=file.filename
            )

        pred_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        label = CLASS_LABELS.get(pred_idx, f"Class {pred_idx}")

        # Lower threshold to handle medium confidence cases
        threshold = 0.50
        if confidence < threshold:
            display_label = "❌ No clear pattern detected"
        else:
            display_label = label.replace('_', ' ').title()

        print(f" Prediction: {display_label} (Confidence: {confidence:.2f})")

        return render_template(
            'index.html',
            filename=file.filename,
            pattern=display_label,
            confidence=f"{confidence:.2f}"
        )

    except Exception as e:
        print("❌ Prediction error:", file=sys.stderr)
        traceback.print_exc()
        return render_template(
            'index.html',
            pattern=f"❌ Prediction failed: {str(e)}",
            confidence="0.00"
        )



# 404 Page

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404



# Run the App

if __name__ == '__main__':
    app.run(debug=True)
