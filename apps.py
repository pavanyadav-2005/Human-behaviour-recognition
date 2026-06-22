from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model("model/final_hybrid_model.keras")

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
SEQUENCE_LENGTH = 10
CLASSES_LIST = ['Clapping', 'Meet and Split', 'Sitting', 'Standing Still', 'Walking', 'Walking While Reading Book', 'Walking While Using Phone']

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, total_frames // SEQUENCE_LENGTH) if total_frames >= SEQUENCE_LENGTH else 1
    
    for i in range(SEQUENCE_LENGTH):
        frame_idx = min(i * frame_skip, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame = frame / 255.0
        frames.append(frame)
    
    cap.release()
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])
    
    return np.array(frames)

def predict_action(video_path):
    frames = extract_frames(video_path)
    if frames is None or len(frames) != SEQUENCE_LENGTH:
        return None, None
    
    input_data = np.expand_dims(frames, axis=0)
    predictions = model.predict(input_data)
    action = CLASSES_LIST[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return action, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    action, confidence = predict_action(file_path)
    if action is None:
        return render_template('index.html', error='Prediction failed')
    
    return render_template('index.html', action=action, confidence=confidence, video_url=file_path)

if __name__ == '__main__':
    app.run(debug=True)
