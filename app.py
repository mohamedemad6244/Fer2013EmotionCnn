from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load trained model
MODEL_PATH = 'saved_models/best_emotion_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Define emotion classes
expressions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Preprocess uploaded image
def preprocess_image(image):
    try:
        image = image.resize((48, 48))
        image = image.convert('L')  # Convert to grayscale
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)       # Add batch dimension
        image = np.expand_dims(image, axis=-1)      # Add channel dimension
        return image
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return None


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/imgDetection', methods=['GET', 'POST'])
def img_detection():
    if request.method == 'GET':
        return render_template('imgDetection/imgDetection.html')

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        if processed_image is None:
            return jsonify({"error": "Invalid image processing"}), 400

        predictions = model.predict(processed_image)
        predicted_index = int(np.argmax(predictions))
        predicted_label = expressions[predicted_index]
        confidence = float(np.max(predictions))

        return render_template(
            'imgDetection/imgDetection.html',
            prediction=predicted_label,
            confidence=confidence
        )

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)
