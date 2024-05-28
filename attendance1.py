import os
import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime

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

# Dictionary to store the last seen time for each recognized face
last_seen_time = {}
appearance_logged = set()  # Set to keep track of logged appearances

def recognize_faces(frame):
    global last_seen_time, appearance_logged

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    recognized_names = []
    now = datetime.now()

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
                last_seen_time[name] = now  # Update last seen time

                if name not in appearance_logged or (now - last_seen_time[name]).total_seconds() > 10:
                    recognized_names.append(name)
                    appearance_logged.add(name)  # Log this appearance

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    # Check if any previously recognized faces have disappeared and reappeared
    names_to_remove = []
    for name, last_seen in list(last_seen_time.items()):
        if (now - last_seen).total_seconds() > 10:  # If not seen for more than 10 seconds
            appearance_logged.discard(name)  # Remove from logged appearances
            names_to_remove.append(name)  # Mark for removal

    for name in names_to_remove:
        del last_seen_time[name]  # Remove from last_seen_time

    return recognized_names, frame

def update_history_file(recognized_names):
    history_file = 'history.txt'
    now = datetime.now()

    # Write new entries to the history file
    with open(history_file, 'a') as file:
        for name in recognized_names:
            if name != "Unknown":
                file.write(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        recognized_names, processed_frame = recognize_faces(frame)
        update_history_file(recognized_names)
        cv2.imshow('Video', processed_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):  # Break the loop if 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
