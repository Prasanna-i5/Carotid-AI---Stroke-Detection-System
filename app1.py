import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops

import tensorflow as tf
from tensorflow.keras.models import model_from_json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------ LOAD SVM ------------------
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ LOAD CNN ------------------
# Requires: pip uninstall tensorflow keras -y && pip install tensorflow==2.13.1
with open("model_dense.json", "r") as json_file:
    json_config = json_file.read()

cnn_model = model_from_json(json_config)
cnn_model.load_weights("model_dense.weights.h5")
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print("✅ CNN model loaded successfully.")

# ------------------ GLCM FEATURE EXTRACTION ------------------
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray,
                        distances=[1, 2],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256,
                        symmetric=True,
                        normed=True)
    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        values = graycoprops(glcm, prop)
        features.extend(values.flatten())
    features.append(np.mean(gray))
    features.append(np.std(gray))
    return np.array(features)

# ------------------ HOME (About Page) ------------------
@app.route('/')
def index():
    return render_template('index.html')

# ------------------ PREDICT PAGE (GET) ------------------
@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

# ------------------ PREDICT (POST — Run Analysis) ------------------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file or file.filename == '':
        return "Error: No file uploaded.", 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    if image is None:
        return "Error: Could not read image file.", 400

    # ----------- SVM -----------
    svm_features = extract_features(image).reshape(1, -1)
    svm_features = scaler.transform(svm_features)
    svm_pred = svm_model.predict(svm_features)[0]
    svm_score = svm_model.decision_function(svm_features)
    svm_label = "Normal" if svm_pred == 0 else "Stroke"
    svm_conf = round(abs(svm_score[0]) * 100, 2)

    # ----------- CNN -----------
    img = cv2.resize(image, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = cnn_model.predict(img)[0]   # [prob_normal, prob_stroke]
    class_index = np.argmax(pred)
    confidence = pred[class_index] * 100
    labels = ["Normal", "Stroke"]
    cnn_label = labels[class_index]
    cnn_conf = round(confidence, 2)

    return render_template('predict.html',
                           image_path=filepath,
                           svm_label=svm_label,
                           svm_conf=svm_conf,
                           cnn_label=cnn_label,
                           cnn_conf=cnn_conf)

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
