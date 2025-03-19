import numpy as np
import cv2

def compute_energy(gray):
    """Computes the energy map using |dI/dx| + |dI/dy| with OpenCV Sobel"""
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y
    energy = np.abs(dx) + np.abs(dy)  # Energy = |dx| + |dy|
    return energy