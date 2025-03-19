import numpy as np

def compute_energy_manual(gray):
    """Manually computes the energy map using |dI/dx| + |dI/dy| with 3x3 Sobel kernels, no OpenCV"""
    height, width = gray.shape
    gray = gray.astype(np.float64)
    
    # Manually create padding with BORDER_REFLECT_101
    gray_padded = np.zeros((height + 2, width + 2), dtype=np.float64)
    
    # Fill the interior with the original image
    gray_padded[1:-1, 1:-1] = gray
    
    # Reflect borders (BORDER_REFLECT_101: mirror without repeating edge)
    gray_padded[0, 1:-1] = gray[1, :]      # Top row
    gray_padded[-1, 1:-1] = gray[-2, :]    # Bottom row
    gray_padded[1:-1, 0] = gray[:, 1]      # Left column
    gray_padded[1:-1, -1] = gray[:, -2]    # Right column
    
    # Corners
    gray_padded[0, 0] = gray[1, 1]         # Top-left
    gray_padded[0, -1] = gray[1, -2]       # Top-right
    gray_padded[-1, 0] = gray[-2, 1]       # Bottom-left
    gray_padded[-1, -1] = gray[-2, -2]     # Bottom-right
    
    # Create output arrays
    dx = np.zeros((height, width), dtype=np.float64)
    dy = np.zeros((height, width), dtype=np.float64)
    
    # Compute gradients over the entire image
    for y in range(height):
        for x in range(width):
            yp, xp = y + 1, x + 1
            # X gradient
            dx[y, x] = (
                -1 * gray_padded[yp-1, xp-1] + 
                 0 * gray_padded[yp-1, xp] + 
                 1 * gray_padded[yp-1, xp+1] + 
                -2 * gray_padded[yp, xp-1] + 
                 0 * gray_padded[yp, xp] + 
                 2 * gray_padded[yp, xp+1] + 
                -1 * gray_padded[yp+1, xp-1] + 
                 0 * gray_padded[yp+1, xp] + 
                 1 * gray_padded[yp+1, xp+1]
            )
            # Y gradient
            dy[y, x] = (
                -1 * gray_padded[yp-1, xp-1] + 
                -2 * gray_padded[yp-1, xp] + 
                -1 * gray_padded[yp-1, xp+1] + 
                 0 * gray_padded[yp, xp-1] + 
                 0 * gray_padded[yp, xp] + 
                 0 * gray_padded[yp, xp+1] + 
                 1 * gray_padded[yp+1, xp-1] + 
                 2 * gray_padded[yp+1, xp] + 
                 1 * gray_padded[yp+1, xp+1]
            )
    
    energy = np.abs(dx) + np.abs(dy)
    return energy