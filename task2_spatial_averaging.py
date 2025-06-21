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

def spatial_average(image, kernel_size_tuple):
    blurred_image = cv2.blur(image, kernel_size_tuple)
    return blurred_image

if __name__ == "__main__":
    print("--- Task 2: Spatial Averaging ---")
    display_image(img_color_original, "Original Color Image")

    kernel_sizes = [(3, 3), (10, 10), (20, 20)]
    for k_size in kernel_sizes:
        blurred_img = spatial_average(img_color_original.copy(), k_size)
        display_image(blurred_img, f"Color Image Blurred with {k_size[0]}x{k_size[1]} Kernel")
    
    print("\nApplying to Grayscale for demonstration:")
    display_image(img_gray_original, "Original Grayscale Image")
    for k_size in kernel_sizes:
        blurred_gray_img = spatial_average(img_gray_original.copy(), k_size)
        display_image(blurred_gray_img, f"Grayscale Image Blurred with {k_size[0]}x{k_size[1]} Kernel")
        
    print("Task 2 Completed.")
