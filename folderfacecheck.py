import os
import shutil
import recognizer
import cv2
import face_recognition


def has_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    return len(face_locations) > 0

def move_images_without_faces(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    total_images = 0
    images_checked = 0

    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1
                image_path = os.path.join(root, filename)
                if not has_face(image_path):
                    relative_path = os.path.relpath(image_path, source_folder)
                    destination_path = os.path.join(destination_folder, relative_path)
                    destination_subfolder = os.path.dirname(destination_path)
                    os.makedirs(destination_subfolder, exist_ok=True)
                    shutil.move(image_path, destination_path)
                    print(f"Moved {filename} to {destination_subfolder}")
                else:
                    images_checked += 1
                    print(f"Image {filename} contains a face.")

    print(f"Total images checked: {total_images}")
    print(f"Images with faces: {images_checked}")
    print(f"Images moved: {total_images - images_checked}")

source_folder = "dataset"
destination_folder = "Noface"

move_images_without_faces(source_folder, destination_folder)
