import os
import cv2
import numpy as np
import face_recognition
import pickle
import shutil
from datetime import datetime

# Function to load known face encodings and names from a pickle file
def load_known_encodings(file_path):
    with open(file_path, 'rb') as file:
        encodings_dict = pickle.load(file)

    known_encodings = []
    known_names = []

    for name, encodings in encodings_dict.items():
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    return known_encodings, known_names

# Function to recognize faces in a video feed
def recognize_faces(frame, known_encodings, known_names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    recognized_names = []

    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_face_image, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_face_image, face_locations)

        name = "Unknown"
        color = (0, 0, 255)  # Red color for unknown

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                color = (0, 255, 0)  # Green color for known

        recognized_names.append(name)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    return recognized_names, frame

# Function to save the history of identified persons in a text file
def save_history(person_name, history_file):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(history_file, "a") as file:
        file.write(f"{timestamp}: {person_name}\n")

# Define paths and parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
pickle_file = "face_encodings.pkl"
history_file = "identification_history.txt"

# Load known face encodings
known_encodings, known_names = load_known_encodings(pickle_file)

# Real-time face recognition
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        recognized_names, processed_frame = recognize_faces(frame, known_encodings, known_names)
        cv2.imshow('Video', processed_frame)

        for name in recognized_names:
            if name != "Unknown":
                save_history(name, history_file)

        key = cv2.waitKey(1)
        if key == ord('q'):  # Break the loop if 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    main()  # Run real-time face recognition
