from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load the pre-trained MobileNetV2 model for image classification
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([
    hub.KerasLayer(model_url, input_shape=(224, 224, 3))
])

# Load the ImageNet labels
LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', LABELS_URL)
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

def prepare_image(file):
    img = Image.open(file)
    img = img.convert('RGB')
    img = img.resize((224, 224))  # MobileNetV2 standard input size
    img = np.array(img, dtype=np.float32)  # Ensure the array is of type float32
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize to [0, 1] range
    return img

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        img = prepare_image(file)

        # Perform inference
        logits = model.predict(img)[0]  # Get the raw logits
        predictions = tf.nn.softmax(logits).numpy()  # Apply softmax to get probabilities
        top_indices = predictions.argsort()[-3:][::-1]  # Get indices of the top 3 predictions

        # Prepare the top 3 predictions
        top_predictions = {}
        for i, index in enumerate(top_indices):
            predicted_label = labels[index]
            confidence_level = round(float(predictions[index]), 5)
            top_predictions[f'prediction_{i+1}'] = {
                'predicted_label': predicted_label,
                'confidence_level': confidence_level
            }

        return jsonify(top_predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, port=8000, host='0.0.0.0')
