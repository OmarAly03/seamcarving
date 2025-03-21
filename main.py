import numpy as np
import matplotlib.pyplot as plt
import cv2  # Needed for OpenCV-based energy computation and image loading
from energy import compute_energy_manual
from image import load_image
from seamcarving import seam_carve

image_path = "lake.jpg"  
num_seams_vertical = 50  
num_seams_horizontal = 50  
seam_carve(image_path, num_seams_vertical, num_seams_horizontal)


