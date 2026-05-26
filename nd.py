import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image

from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- FLASK APP ---------------- #

app = Flask(__name__)

app.secret_key = "supersecretkey"

# ---------------- CONFIG ---------------- #

UPLOAD_FOLDER = 'static/uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'png'}

MODEL_PATH = "models/tsynet_ortho_predictor.h5"

IMAGE_SIZE = (128, 128)

# Your actual folder names
class_names = ["normal", "overbite", "underbite"]

# ---------------- LOAD MODEL ---------------- #

model = load_model(MODEL_PATH, compile=False)

# ---------------- FILE VALIDATION ---------------- #

def allowed_file(filename):

    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ---------------- IMAGE PREPROCESSING ---------------- #

def prepare_image(image_path):

    # Load grayscale image
    img = load_img(
        image_path,
        color_mode='grayscale',
        target_size=IMAGE_SIZE
    )

    # Convert to array
    img_array = img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ---------------- XRAY VALIDATION ---------------- #

def is_xray_image(image_path):

    img = Image.open(image_path).convert("RGB")

    img_array = np.array(img)

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    avg_diff = (diff_rg + diff_rb + diff_gb) / 3

    return avg_diff < 15

# ---------------- HOME PAGE ---------------- #

@app.route('/')
def home():

    return render_template('home.html')

# ---------------- DASHBOARD ---------------- #

@app.route('/dashboard', methods=['GET', 'POST'])

def dashboard():

    if request.method == 'POST':

        # Check file uploaded
        if 'image' not in request.files:

            flash("No image uploaded")

            return redirect(request.url)

        file = request.files['image']

        # Check filename
        if file.filename == '':

            flash("No selected file")

            return redirect(request.url)

        # Validate extension
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename
            )

            file.save(filepath)

            
            

            # Check if valid X-ray
            if not is_xray_image(filepath):

                flash("Please upload a valid dental X-ray image")

                return redirect(request.url)

            # Prepare image
            img_array = prepare_image(filepath)

            # Predict
            preds = model.predict(img_array)[0]

            print("Raw Predictions:", preds)

            # Get predicted class index
            class_index = np.argmax(preds)

            print("Predicted Index:", class_index)

            # Get class name
            class_name = class_names[class_index]

            print("Class Name:", class_name)

            # Get confidence
            confidence = np.max(preds)

            print("Confidence:", confidence)
            

            # Show result
            return render_template(
                'dashboard.html',
                filename=filename,
                prediction=class_name,
                confidence=round(confidence * 100, 2)
            )

        else:

            flash("Allowed image types: bmp, jpg, jpeg, png")

            return redirect(request.url)

    return render_template('dashboard.html')

# ---------------- DISPLAY UPLOADED IMAGE ---------------- #

@app.route('/uploads/<filename>')

def uploaded_file(filename):

    return redirect(
        url_for(
            'static',
            filename='uploads/' + filename
        )
    )

# ---------------- RUN APP ---------------- #

if __name__ == "__main__":

    app.run(debug=True)