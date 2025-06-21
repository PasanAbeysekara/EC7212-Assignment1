import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

IMAGE_PATH = 'sample_image.jpg'

def display_image(image, title="Image", cmap=None):
    plt.figure(figsize=(6, 6))
    if cmap:
        plt.imshow(image, cmap=cmap)
    elif len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show(block=True)

try:
    img_color_original = cv2.imread(IMAGE_PATH)
    if img_color_original is None:
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    if angle % 90 == 0:
        if angle == 90 or angle == -270:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180 or angle == -180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270 or angle == -90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle % 360 == 0:
            return image.copy()
        else: # Should not happen with angle % 90 == 0
            return image.copy() 
    else:
        rad = math.radians(angle)
        sin_a = abs(math.sin(rad))
        cos_a = abs(math.cos(rad))

        new_width = int((height * sin_a) + (width * cos_a))
        new_height = int((height * cos_a) + (width * sin_a))

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]

        rotated_image = cv2.warpAffine(image, M, (new_width, new_height), flags=cv2.INTER_LINEAR)
        return rotated_image

if __name__ == "__main__":
    print("--- Task 3: Rotate Image ---")
    display_image(img_color_original, "Original Color Image")

    angles_to_rotate = [45, 90]
    for ang in angles_to_rotate:
        rotated_img = rotate_image(img_color_original.copy(), ang)
        display_image(rotated_img, f"Color Image Rotated by {ang} Degrees")
        
    print("Task 3 Completed.")
