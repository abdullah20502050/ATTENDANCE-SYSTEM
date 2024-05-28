import cv2
import numpy as np
import os

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def flip_image(image):
    return cv2.flip(image, 1)

def scale_image(image, scale_factor):
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

def translate_image(image, x_shift, y_shift):
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

def adjust_brightness(image, brightness_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)
    adjusted_hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, contrast_factor):
    return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

def add_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_blur(image, blur_amount):
    return cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]

input_dir = "dataset"
output_dir = "Augmented imgaes"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rotation_angle = 15
flip_horizontal = True
scale_factor = 1.2
x_shift, y_shift = 20, -10
brightness_factor = 1.2
contrast_factor = 1.5
noise_level = 25
blur_amount = 5
crop_x, crop_y, crop_width, crop_height = 50, 50, 150, 150

for sub_dir in os.listdir(input_dir):
    sub_dir_path = os.path.join(input_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        output_sub_dir = os.path.join(output_dir, sub_dir)
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        for filename in os.listdir(sub_dir_path):
            image_path = os.path.join(sub_dir_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue  # Skip to the next image

            augmented_images = [
                rotate_image(image, rotation_angle),
                flip_image(image) if flip_horizontal else image,
                scale_image(image, scale_factor),
                translate_image(image, x_shift, y_shift),
                adjust_brightness(image, brightness_factor),
                adjust_contrast(image, contrast_factor),
                add_noise(image, noise_level),
                apply_blur(image, blur_amount),
                crop_image(image, crop_x, crop_y, crop_width, crop_height)
            ]

            for i, aug_img in enumerate(augmented_images):
                output_filename = f"{os.path.splitext(filename)[0]}_aug{i}.png"
                output_path = os.path.join(output_sub_dir, output_filename)
                if not cv2.imwrite(output_path, aug_img):
                    print(f"Failed to write image: {output_path}")

print("Data augmentation complete.")
