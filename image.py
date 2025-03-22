import cv2

def load_image(image_path):
    """Loads the image and converts it to grayscale"""
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found or cannot be opened.")
        exit()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray