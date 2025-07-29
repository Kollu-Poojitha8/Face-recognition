import cv2
import numpy as np
import time
import os
import urllib.request
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
import gdown
import dlib

def download_cascade_classifier():
    # Download face cascade
    face_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    face_cascade_path = "haarcascade_frontalface_default.xml"
    
    if not os.path.exists(face_cascade_path):
        print("Downloading face detection model...")
        urllib.request.urlretrieve(face_cascade_url, face_cascade_path)
        print("Download completed!")
    
    # Download eye cascade
    eye_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    eye_cascade_path = "haarcascade_eye.xml"
    
    if not os.path.exists(eye_cascade_path):
        print("Downloading eye detection model...")
        urllib.request.urlretrieve(eye_cascade_url, eye_cascade_path)
        print("Download completed!")
    
    return face_cascade_path, eye_cascade_path

def download_emotion_model():
    model_path = "emotion_model.h5"
    if not os.path.exists(model_path):
        print("Downloading pre-trained emotion detection model...")
        # Download pre-trained model from Google Drive
        url = 'https://drive.google.com/uc?id=1-6U-AHcF3XQN1P1P1P1P1P1P1P1P1P1P1'
        gdown.download(url, model_path, quiet=False)
        print("Model downloaded and saved!")
    return model_path

def download_landmark_model():
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print("Downloading facial landmark model...")
        url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
        import bz2
        with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", "rb") as source, open(model_path, "wb") as dest:
            dest.write(source.read())
        print("Facial landmark model downloaded and extracted!")
    return model_path

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def detect_emotion(landmarks, frame, face_roi):
    # Extract key facial points
    # Mouth corners and center
    left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
    right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
    mouth_center = (landmarks.part(66).x, landmarks.part(66).y)
    mouth_top = (landmarks.part(62).x, landmarks.part(62).y)
    mouth_bottom = (landmarks.part(66).x, landmarks.part(66).y)
    
    # Eye corners
    left_eye_left = (landmarks.part(36).x, landmarks.part(36).y)
    left_eye_right = (landmarks.part(39).x, landmarks.part(39).y)
    right_eye_left = (landmarks.part(42).x, landmarks.part(42).y)
    right_eye_right = (landmarks.part(45).x, landmarks.part(45).y)
    
    # Eyebrows
    left_eyebrow = (landmarks.part(19).x, landmarks.part(19).y)
    right_eyebrow = (landmarks.part(24).x, landmarks.part(24).y)
    
    # Calculate features
    mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
    mouth_height = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))
    eye_width = np.linalg.norm(np.array(left_eye_left) - np.array(left_eye_right))
    eyebrow_height = np.linalg.norm(np.array(left_eyebrow) - np.array(right_eyebrow))
    
    # Calculate mouth corner angles
    left_mouth_angle = calculate_angle(left_mouth, mouth_center, (left_mouth[0], mouth_center[1]))
    right_mouth_angle = calculate_angle(right_mouth, mouth_center, (right_mouth[0], mouth_center[1]))
    
    # Calculate eyebrow angles
    left_eyebrow_angle = calculate_angle(left_eyebrow, (left_eyebrow[0], left_eye_left[1]), left_eye_left)
    right_eyebrow_angle = calculate_angle(right_eyebrow, (right_eyebrow[0], right_eye_right[1]), right_eye_right)
    
    # Draw facial features for debugging
    cv2.line(frame, left_mouth, mouth_center, (0, 255, 0), 1)
    cv2.line(frame, right_mouth, mouth_center, (0, 255, 0), 1)
    cv2.line(frame, mouth_top, mouth_bottom, (0, 255, 0), 1)
    
    # Emotion detection based on facial features
    # Happy (smile with visible teeth)
    if (mouth_height > eye_width * 0.4 and  # Open mouth
        left_mouth_angle > 15 and right_mouth_angle > 15 and  # Mouth corners up
        mouth_width > eye_width * 1.2):  # Wide smile
        return "Happy"
    
    # Sad
    elif (mouth_height < eye_width * 0.2 and  # Small mouth
          left_mouth_angle < -5 and right_mouth_angle < -5 and  # Mouth corners down
          left_eyebrow_angle < 20 and right_eyebrow_angle < 20):  # Drooping eyebrows
        return "Sad"
    
    # Angry
    elif (left_eyebrow_angle < 15 and right_eyebrow_angle < 15 and  # Eyebrows down
          mouth_height < eye_width * 0.2 and  # Tight mouth
          eyebrow_height < eye_width * 0.6):  # Eyebrows close together
        return "Angry"
    
    # Fearful
    elif (left_eyebrow_angle > 35 and right_eyebrow_angle > 35 and  # Eyebrows raised
          mouth_height > eye_width * 0.3 and  # Open mouth
          eyebrow_height > eye_width * 0.7):  # Eyebrows far apart
        return "Fearful"
    
    # Surprised
    elif (left_eyebrow_angle > 40 and right_eyebrow_angle > 40 and  # Eyebrows very high
          mouth_height > eye_width * 0.5 and  # Very open mouth
          mouth_width > eye_width * 1.3):  # Wide mouth
        return "Surprised"
    
    else:
        return "Neutral"

def detect_face_movement():
    # Download and get the cascade classifier paths
    face_cascade_path, eye_cascade_path = download_cascade_classifier()
    
    # Download and load facial landmark model
    landmark_model_path = download_landmark_model()
    predictor = dlib.shape_predictor(landmark_model_path)
    
    # Initialize the webcam with AVFoundation backend
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Please check if your camera is connected and you have granted permission to access it.")
        return
        
    # Load the cascade classifiers
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    if face_cascade.empty() or eye_cascade.empty():
        print("Error: Failed to load cascade classifiers")
        return
    
    # Initialize variables for movement detection
    prev_face_center = None
    movement_threshold = 10  # Reduced threshold to detect even slight movements
    last_direction = None
    last_direction_time = time.time()
    direction_cooldown = 0.1  # Reduced cooldown for more responsive detection
    
    # Get frame dimensions for center calculation
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera")
        return
    
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2
    
    print("Camera initialized successfully!")
    print("Press 'q' to quit the program")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
            
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Calculate face center
            face_center = (x + w//2, y + h//2)
            
            # Draw center point
            cv2.circle(frame, face_center, 5, (0, 255, 0), -1)
            
            # Draw frame center point
            cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)
            
            # Detect eyes in the face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Check for glasses
            glasses_status = "Glasses: OFF"
            if len(eyes) == 0:  # If no eyes detected, likely wearing glasses
                glasses_status = "Glasses: ON"
            
            # Detect emotion using facial landmarks
            try:
                # Convert face region to dlib rectangle
                rect = dlib.rectangle(x, y, x+w, y+h)
                landmarks = predictor(gray, rect)
                
                # Draw facial landmarks
                for n in range(68):
                    x_point = landmarks.part(n).x
                    y_point = landmarks.part(n).y
                    cv2.circle(frame, (x_point, y_point), 1, (0, 255, 255), -1)
                
                # Detect emotion
                emotion = detect_emotion(landmarks, frame, roi_color)
            except:
                emotion = "Unknown"
            
            # Detect movement if we have a previous position
            if prev_face_center is not None:
                dx_movement = face_center[0] - prev_face_center[0]
                dy_movement = face_center[1] - prev_face_center[1]
                
                # Only update direction if enough time has passed
                current_time = time.time()
                if current_time - last_direction_time > direction_cooldown:
                    # Check for horizontal movement first
                    if abs(dx_movement) > movement_threshold:
                        if dx_movement < 0:
                            direction = "LEFT"
                        else:
                            direction = "RIGHT"
                    # Then check for vertical movement
                    elif abs(dy_movement) > movement_threshold:
                        if dy_movement < 0:
                            direction = "UP"
                        else:
                            direction = "DOWN"
                    
                    # Display current direction, glasses status, and emotion
                    if 'direction' in locals():
                        cv2.putText(frame, f"Direction: {direction}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, glasses_status, (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Emotion: {emotion}", (10, 110),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        last_direction_time = current_time
            
            prev_face_center = face_center
        
        # Display the frame
        cv2.imshow('Face Movement Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face_movement()
