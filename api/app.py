from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import time
import io
from PIL import Image
import socket

POD = socket.gethostname() 

app = Flask(__name__)

model = MobileNetV2(weights="imagenet")

@app.route('/classify', methods=['POST'])
def classify_image():
    start = time.time()

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]

    result = {
        "prediction": decoded[1],
        "confidence": float(decoded[2]),
        "processing_ms": int((time.time() - start) * 1000),
        "served_by": POD
    }
    return jsonify(result)

@app.route('/')
def home():
    return "Image classifier running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
