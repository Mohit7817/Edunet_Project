from flask import Flask, render_template, request
import os
import numpy as np
import csv
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('model/garbage_classifier_model.h5')

# Class labels (update as per your model)
class_names = ['Organic', 'Recyclable', 'Hazardous', 'General']

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in request.'

    file = request.files['file']
    if file.filename == '':
        return 'No file selected.'

    # Save uploaded image
    upload_folder = 'static/uploaded'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Log prediction to CSV
    log_path = 'prediction_log.csv'
    with open(log_path, mode='a', newline='') as log_file:
        writer = csv.writer(log_file)
        if os.stat(log_path).st_size == 0:
            writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            file.filename,
            predicted_class,
            f"{confidence:.2f}"
        ])

    # Render result
    rendered_page = render_template(
        'index.html',
        prediction=predicted_class,
        confidence=round(confidence, 2),
        image_path=file_path
    )

    # Auto-delete uploaded image
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")

    return rendered_page

if __name__ == '__main__':
    app.run(debug=True)
