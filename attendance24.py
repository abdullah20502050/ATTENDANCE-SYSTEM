import os
import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime, timedelta

# Path to the main folder containing subfolders of different persons
main_folder = "dataset"

# This dictionary will store the face encodings along with their corresponding labels
encodings_dict = {}

for person_name in os.listdir(main_folder):
    person_path = os.path.join(main_folder, person_name)
    # Initialize the list for storing encodings for the person
    person_encodings = []
    
    print(f"Processing images for {person_name}...")
    image_count = 0  # To count number of images processed for this person
    
    for filename in os.listdir(person_path):
        image_path = os.path.join(person_path, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Encode the face from the image
        encodings = face_recognition.face_encodings(image)
        # Add encodings to the list
        person_encodings.extend(encodings)
        
        image_count += 1
        print(f"Encoded {image_count} images for {person_name}.")

    if person_encodings:
        encodings_dict[person_name] = person_encodings
        print(f"Encodings for {person_name} added with {len(person_encodings)} encodings.")

# Save the encodings dictionary to a pickle file
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump(encodings_dict, f)

print("All face encodings have been saved to face_encodings.pkl.")

# Load known face encodings and names from a pickle file
with open('face_encodings.pkl', 'rb') as file:
    encodings_dict = pickle.load(file)

known_encodings = []
known_names = []

# Iterate through the dictionary to populate lists
for name, encodings in encodings_dict.items():
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

# Setup Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Failed to load Haar Cascade classifier.")

def recognize_faces(frame):
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

def update_history_file(recognized_names, session_entries):
    history_file = 'history.txt'
    now = datetime.now()

    # Read existing entries
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            lines = file.readlines()
    else:
        lines = []

    # Filter out entries older than 24 hours
    updated_lines = []
    for line in lines:
        entry_name, entry_time = line.strip().split(',')
        entry_time = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
        if now - entry_time <= timedelta(hours=24):
            updated_lines.append(line)

    # Add new entries
    for name in recognized_names:
        if name == "Unknown":
            continue
        if name not in session_entries:
            updated_lines.append(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            session_entries.add(name)

    # Write updated entries back to the file
    with open(history_file, 'w') as file:
        file.writelines(updated_lines)

def main():
    cap = cv2.VideoCapture(0)
    session_entries = set()  # Track entries in the current session

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        recognized_names, processed_frame = recognize_faces(frame)
        update_history_file(recognized_names, session_entries)
        cv2.imshow('Video', processed_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):  # Break the loop if 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
