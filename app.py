from flask import Flask, request, jsonify, render_template
from functions import load_model, predict_image, get_class_name
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = load_model('modelo.h5')

# Directorio donde se guardarán las imágenes subidas
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('home.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    prediction = predict_image(model, file_path)
    animal_name = get_class_name(prediction) 

    os.remove(file_path)

    return jsonify({'prediction': animal_name})


if __name__ == '__main__':
    app.run(debug=True)




