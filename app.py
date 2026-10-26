
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import os

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "animal_species_model.tflite"
)

CLASSES = ["ELEPHANT", "HORSE", "LION", "cat", "dog"]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Animal Species Recognition API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image"}), 400

    try:
        image = Image.open(request.files["image"]).convert("RGB")
        image = image.resize((224, 224))
        data = np.asarray(image, dtype=np.float32)
        data = np.expand_dims(data, axis=0)

        input_info = input_details[0]
        if input_info["dtype"] == np.uint8:
            scale, zero = input_info["quantization"]
            if scale:
                data = np.clip(
                    np.round(data / scale + zero), 0, 255
                ).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        elif input_info["dtype"] == np.int8:
            scale, zero = input_info["quantization"]
            data = np.round(data / scale + zero)
            data = np.clip(data, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_info["index"], data)
        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        index = int(np.argmax(output))
        animal = CLASSES[index]

        return jsonify({
            "animal": animal.title(),
            "category": (
                "Pet" if animal.lower() in ["cat", "dog"]
                else "Domestic Animal" if animal == "HORSE"
                else "Wild Animal"
            ),
            "confidence": round(float(output[index]) * 100, 2)
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500
