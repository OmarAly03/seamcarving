import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from numba import njit
from energy import compute_energy_manual
from image import load_image

@njit
def find_seam_kernel(energy, h, w):
    """Inner loop for finding seam, optimized with Numba"""
    M = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)
    
    for i in range(1, h):
        for j in range(w):
            left = M[i-1, j-1] if j > 0 else np.inf
            up = M[i-1, j]
            right = M[i-1, j+1] if j < w - 1 else np.inf
            
            min_val = min(left, up, right)
            M[i, j] += min_val
            
            if min_val == left:
                backtrack[i, j] = j - 1
            elif min_val == up:
                backtrack[i, j] = j
            else:
                backtrack[i, j] = j + 1
    
    return backtrack

def find_seam(energy):
    """Finds the optimal seam to remove using Dynamic Programming"""
    h, w = energy.shape
    backtrack = find_seam_kernel(energy, h, w)
    return backtrack

def mark_seam(img, backtrack):
    """Marks the seam in red without removing it"""
    h, w, _ = img.shape
    seam_img = img.copy()
    
    j = np.argmin(backtrack[-1])
    for i in range(h-1, -1, -1):
        seam_img[i, j] = [0, 0, 255]
        j = backtrack[i, j]
    
    return seam_img

def remove_seam(img, backtrack):
    """Removes a seam from the image"""
    h, w, _ = img.shape
    mask = np.ones((h, w), dtype=np.bool_)
    
    j = np.argmin(backtrack[-1])
    for i in range(h-1, -1, -1):
        mask[i, j] = False
        j = backtrack[i, j]
    
    return img[mask].reshape((h, w - 1, 3))

def seam_carve(image_path, num_seams_vertical=0, num_seams_horizontal=0):
    """Removes vertical and horizontal seams"""
    start_time = time.time()
    
    img, gray = load_image(image_path)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    marked_img = img.copy()
    
    for i in range(num_seams_vertical):
        energy = compute_energy_manual(gray)
        backtrack = find_seam(energy)
        marked_img = mark_seam(marked_img, backtrack)
        gray = cv2.cvtColor(marked_img, cv2.COLOR_BGR2GRAY)
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
    plt.title('Vertical Seams Marked')
    plt.axis('off')
    
    img, gray = load_image(image_path)
    
    for i in range(num_seams_vertical):
        energy = compute_energy_manual(gray)
        backtrack = find_seam(energy)
        img = remove_seam(img, backtrack)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(num_seams_horizontal):
        energy = compute_energy_manual(gray)
        backtrack = find_seam(energy)
        img = remove_seam(img, backtrack)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    total_time = time.time() - start_time
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Final Image ({num_seams_vertical}v, {num_seams_horizontal}h seams removed)\nTotal time = {total_time:.4f} seconds')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('numba_approach', dpi=200)
    plt.show()
    
    print(f"Total Processing Time: {total_time:.4f} seconds")
    
    return img, total_time