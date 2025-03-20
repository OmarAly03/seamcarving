import numpy as np
import matplotlib.pyplot as plt
import cv2
from energy import compute_energy_manual
from image import load_image
from seamcarving import seam_carve

image_path = "mountain.jpg"
num_seams_vertical = 200
num_seams_horizontal = 200
seam_carve(image_path, num_seams_vertical, num_seams_horizontal)