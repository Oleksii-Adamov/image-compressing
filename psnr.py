import numpy as np


def psnr(original_image, compressed_image):
    mse = np.mean((original_image - compressed_image) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)