import os
import face_recognition
import pickle

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
