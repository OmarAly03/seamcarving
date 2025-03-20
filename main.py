import numpy as np
import matplotlib.pyplot as plt
import cv2  # Needed for OpenCV-based energy computation and image loading
from energy import compute_energy_manual
from image import load_image
from seamcarving import seam_carve

# Run the algorithm
image_path = "lake.jpg"  # Change this to your image path
num_seams_vertical = 50  # Number of vertical seams to remove
num_seams_horizontal = 50  # Number of horizontal seams to remove
seam_carve(image_path, num_seams_vertical, num_seams_horizontal)


