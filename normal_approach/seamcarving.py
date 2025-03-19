import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from energy import compute_energy_manual
from image import load_image

def find_seam(energy):
    """Finds the optimal seam to remove using Dynamic Programming"""
    h, w = energy.shape
    M = energy.copy()  # Cost map
    backtrack = np.zeros_like(M, dtype=np.int32)  # Stores seam path

    # DP initialization
    for i in range(1, h):
        for j in range(w):
            left = M[i-1, j-1] if j > 0 else float('inf')
            up = M[i-1, j]
            right = M[i-1, j+1] if j < w - 1 else float('inf')

            min_val = min(left, up, right)
            M[i, j] += min_val

            # Store the backtrack path
            if min_val == left:
                backtrack[i, j] = j - 1
            elif min_val == up:
                backtrack[i, j] = j
            else:
                backtrack[i, j] = j + 1

    return backtrack

def mark_seam(img, backtrack):
    """Marks the seam in red without removing it"""
    h, w, _ = img.shape
    seam_img = img.copy()
    
    # Find the position of the smallest seam at the bottom row
    j = np.argmin(backtrack[-1])
    for i in range(h-1, -1, -1):  # Backtrack from bottom to top
        seam_img[i, j] = [0, 0, 255]  # Color pixel in red
        j = backtrack[i, j]

    return seam_img

def remove_seam(img, backtrack):
    """Removes a seam from the image"""
    h, w, _ = img.shape
    mask = np.ones((h, w), dtype=np.bool_)

    # Find the position of the smallest seam at the bottom row
    j = np.argmin(backtrack[-1])
    for i in range(h-1, -1, -1):  # Backtrack from bottom to top
        mask[i, j] = False
        j = backtrack[i, j]

    return img[mask].reshape((h, w - 1, 3))

def seam_carve(image_path, num_seams_vertical=0, num_seams_horizontal=0):
    """Removes vertical and horizontal seams"""
    # Start total timing
    start_time = time.time()
    
    # Load image
    img, gray = load_image(image_path)
    
    # Create a figure to show results
    plt.figure(figsize=(15, 8))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Process vertical seams for marking
    marked_img = img.copy()
    
    for i in range(num_seams_vertical):
        # Compute energy
        energy = compute_energy_manual(gray)
        
        # Find seam
        backtrack = find_seam(energy)
        
        # Mark seam
        marked_img = mark_seam(marked_img, backtrack)
        
        # Update grayscale for next iteration
        gray = cv2.cvtColor(marked_img, cv2.COLOR_BGR2GRAY)
    
    # Plot image with seams marked
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
    plt.title('Vertical Seams Marked')
    plt.axis('off')
    
    # Reload original image to remove seams
    img, gray = load_image(image_path)
    
    # Remove vertical seams
    for i in range(num_seams_vertical):
        # Compute energy
        energy = compute_energy_manual(gray)
        
        # Find seam
        backtrack = find_seam(energy)
        
        # Remove seam
        img = remove_seam(img, backtrack)
        
        # Update grayscale for next iteration
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove horizontal seams (rotate image, process as vertical, rotate back)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(num_seams_horizontal):
        # Compute energy
        energy = compute_energy_manual(gray)
        
        # Find seam
        backtrack = find_seam(energy)
        
        # Remove seam
        img = remove_seam(img, backtrack)
        
        # Update grayscale for next iteration
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Plot final image
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Final Image ({num_seams_vertical}v, {num_seams_horizontal}h seams removed)')
    plt.axis('off')
    
    # Adjust layout and display
    plt.tight_layout()

    plt.savefig('output', dpi=200)

    plt.show()
    
    # Print timing information to console
    print(f"Total Processing Time: {total_time:.4f} seconds")
    
    return img, total_time