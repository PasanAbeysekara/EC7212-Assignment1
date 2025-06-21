import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    img_gray_original = cv2.cvtColor(img_color_original, cv2.COLOR_BGR2GRAY)
except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

def reduce_intensity_levels(image, num_levels):
    if not (num_levels > 1 and (num_levels & (num_levels - 1) == 0) and num_levels <= 256):
        raise ValueError("Number of levels must be a power of 2 between 2 and 256.")
    
    image_float = image.astype(float)
    if num_levels == 1: # Avoid division by zero for N-1 if num_levels is 1 (though constraint is >1)
         output_image = np.zeros_like(image_float)
    else:
        output_image = np.floor(image_float * (num_levels / 256.0)) * (255.0 / (num_levels - 1.0))
    
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return output_image

if __name__ == "__main__":
    print("--- Task 1: Reduce Intensity Levels ---")
    display_image(img_gray_original, "Original Grayscale Image")
    
    desired_levels = [128, 64, 32, 16, 8, 4, 2]
    for levels in desired_levels:
        try:
            reduced_img = reduce_intensity_levels(img_gray_original.copy(), levels)
            display_image(reduced_img, f"Grayscale Image with {levels} Intensity Levels")
        except ValueError as e:
            print(e)
    print("Task 1 Completed.")
