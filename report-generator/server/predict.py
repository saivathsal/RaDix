import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from skimage.transform import resize

# Initialize Flask App
app = Flask(__name__)

# Suppress TensorFlow warnings and disable oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model
MODEL_PATH = r"C:\Report Generator\report-generator\server\efficientnet_model.h5"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """ Preprocess the uploaded image """
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image format.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, (227, 227, 3))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img

@app.route("/predict", methods=["POST"])  # Changed to /predict as per your Node.js route
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Preprocess the image
            img = preprocess_image(file)

            # Predict using the model
            predictions = model.predict(img)
            class_names = ['NORMAL', 'BACTERIAL PNEUMONIA', 'VIRAL PNEUMONIA']
            predicted_class = class_names[np.argmax(predictions)]
            confidence = float(np.max(predictions))  # Convert to standard Python float

            # Return response
            response = {
                "prediction": {
                    "class": predicted_class,
                    "confidence": confidence
                }
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)

