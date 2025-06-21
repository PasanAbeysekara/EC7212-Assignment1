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

def block_average_resolution(image, block_size_val):
    height, width = image.shape[:2]
    output_image = image.copy()

    is_color = len(image.shape) == 3

    for r_idx in range(0, height - height % block_size_val, block_size_val):
        for c_idx in range(0, width - width % block_size_val, block_size_val):
            if is_color:
                block = image[r_idx:r_idx+block_size_val, c_idx:c_idx+block_size_val, :]
                if block.size > 0:
                    avg_b = np.mean(block[:,:,0])
                    avg_g = np.mean(block[:,:,1])
                    avg_r = np.mean(block[:,:,2])
                    output_image[r_idx:r_idx+block_size_val, c_idx:c_idx+block_size_val, 0] = avg_b
                    output_image[r_idx:r_idx+block_size_val, c_idx:c_idx+block_size_val, 1] = avg_g
                    output_image[r_idx:r_idx+block_size_val, c_idx:c_idx+block_size_val, 2] = avg_r
            else: # Grayscale
                block = image[r_idx:r_idx+block_size_val, c_idx:c_idx+block_size_val]
                if block.size > 0:
                    average = np.mean(block)
                    output_image[r_idx:r_idx+block_size_val, c_idx:c_idx+block_size_val] = average
                
    return output_image.astype(np.uint8)

if __name__ == "__main__":
    print("--- Task 4: Block Averaging (Spatial Resolution Reduction) ---")
    display_image(img_gray_original, "Original Grayscale Image")
    
    block_sizes_list = [3, 5, 7]
    for b_size in block_sizes_list:
        res_reduced_img = block_average_resolution(img_gray_original.copy(), b_size)
        display_image(res_reduced_img, f"Grayscale Image with {b_size}x{b_size} Block Averaging")

    print("\nApplying to Color for demonstration:")
    display_image(img_color_original, "Original Color Image")
    for b_size in block_sizes_list:
        res_reduced_color_img = block_average_resolution(img_color_original.copy(), b_size)
        display_image(res_reduced_color_img, f"Color Image with {b_size}x{b_size} Block Averaging")
        
    print("Task 4 Completed.")
