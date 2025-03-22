import numpy as np
from numba import njit

@njit
def compute_energy_manual_kernel(gray_padded, height, width):
    """Inner loop for computing energy, optimized with Numba"""
    dx = np.zeros((height, width), dtype=np.float64)
    dy = np.zeros((height, width), dtype=np.float64)
    
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
    
    return dx, dy

def compute_energy_manual(gray):
    """Computes the energy map using |dI/dx| + |dI/dy| with 3x3 Sobel kernels"""
    height, width = gray.shape
    gray = gray.astype(np.float64)
    
    # Manually creating padding with BORDER_REFLECT_101
    gray_padded = np.zeros((height + 2, width + 2), dtype=np.float64)
    gray_padded[1:-1, 1:-1] = gray
    gray_padded[0, 1:-1] = gray[1, :]
    gray_padded[-1, 1:-1] = gray[-2, :]
    gray_padded[1:-1, 0] = gray[:, 1]
    gray_padded[1:-1, -1] = gray[:, -2]
    gray_padded[0, 0] = gray[1, 1]
    gray_padded[0, -1] = gray[1, -2]
    gray_padded[-1, 0] = gray[-2, 1]
    gray_padded[-1, -1] = gray[-2, -2]
    
    # Computing gradients using Numba-optimized kernel
    dx, dy = compute_energy_manual_kernel(gray_padded, height, width)
    
    energy = np.abs(dx) + np.abs(dy)
    return energy