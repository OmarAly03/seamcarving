import numpy as np
from image import load_image
from seamcarving import seam_carve

# Run the algorithm
image_path = "opencv_approach/lake.jpg"  # Change this to your image path
num_seams_vertical = 50  # Number of vertical seams to remove
num_seams_horizontal = 50  # Number of horizontal seams to remove
seam_carve(image_path, num_seams_vertical, num_seams_horizontal)


