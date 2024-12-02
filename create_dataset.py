import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Face Detection.
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up face detection model.
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

DATA_DIR = './data4'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect faces.
        results = face_detection.process(img_rgb)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                x_min = bbox.xmin
                y_min = bbox.ymin
                width = bbox.width
                height = bbox.height
                
                # Store the coordinates of the bounding box.
                x_.extend([x_min, x_min + width])
                y_.extend([y_min, y_min + height])

                # Normalize and collect data for each face.
                data_aux.extend([x_min, y_min, width, height])
            
            if len(data_aux) == 4:
                data.append(data_aux)
                labels.append(dir_)

# Save the face data and labels to a pickle file.
with open('data4_faces.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
