# Imports:
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub # For accessing a pre-trained model.

# Creating Flask application instance:
app = Flask(__name__)

# Loading the pre-trained MobileNetV2 model for image classification:
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([
    hub.KerasLayer(model_url, input_shape=(224, 224, 3))
])

# Loading the ImageNet labels:
labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', labels_url)
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

"""
Image Pre-Processing:
- Input: image file
- Ouput: processed image file
- Description:
    - Takes a raw image, and performs operations such as converting to RGB format, resizing, etc. on the image.
    - The processed image is ready to be sent to a trained model as an input.
"""
def prepare_image(file):
    img = Image.open(file)
    img = img.convert('RGB')
    img = img.resize((224, 224)) # 224x224 is the default input size of the of the MobileNetV2 model.
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0) # Adding a batch dimension to the image array.
    img = img / 255.0 # Normalisng the pixel values to a range [0,1].

    return img

"""
Defining /upload API endpoint
- Input: None
- Ouput: JSON object containing image classification data (top three predited classes with their confidence levels).
- Description:
    - Defines an API endpoint to handle uploads.
    - Runs inference using a pre-trained model and obtains the top 3 predictions made my the model.
    - Returns the predicted classes along with their confidence levels as a JSON object.
    - Performs necessary exception handling and returns appropriate error messages.
"""
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        img = prepare_image(file)

        # Performing inference using the pre-trained model:
        logits = model.predict(img)[0]  # Getting raw logits from model's predictions.
        predictions = tf.nn.softmax(logits).numpy()  # Applying softmax to get probabilities.
        top_indices = predictions.argsort()[-3:][::-1]  # Getting indices of the top 3 predictions.

        # Preparing the top 3 predictions:
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
    app.run(debug=False, port=5000, host='0.0.0.0')
