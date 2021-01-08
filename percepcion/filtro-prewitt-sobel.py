import scipy.ndimage.filters as sc
import numpy as np
import matplotlib.pyplot as plt
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
img = plt.imread(THIS_FOLDER + "/img/fruta.jpg").astype('float') / 255.0

# PREWITT KERNELS
prewitt_x = np.array([[[1.0, 1.0, 1.0],
                       [0.0, 0.0, 0.0],
                       [-1.0, -1.0, -1.0]],
                      [[1.0, 1.0, 1.0],
                       [0.0, 0.0, 0.0],
                       [-1.0, -1.0, -1.0]],
                      [[1.0, 1.0, 1.0],
                       [0.0, 0.0, 0.0],
                       [-1.0, -1.0, -1.0]]])

prewitt_y = np.array([[[1.0, 0.0, -1.0],
                       [1.0, 0.0, -1.0],
                       [1.0, 0.0, -1.0]],
                      [[1.0, 0.0, -1.0],
                       [1.0, 0.0, -1.0],
                       [1.0, 0.0, -1.0]],
                      [[1.0, 0.0, -1.0],
                       [1.0, 0.0, -1.0],
                       [1.0, 0.0, -1.0]]])

sobel_x = np.array([[[1.0, 2.0, 1.0],
                     [0.0, 0.0, 0.0],
                     [-1.0, -2.0, -1.0]],
                    [[1.0, 2.0, 1.0],
                     [0.0, 0.0, 0.0],
                     [-1.0, -2.0, -1.0]],
                    [[1.0, 2.0, 1.0],
                     [0.0, 0.0, 0.0],
                     [-1.0, -2.0, -1.0]]])

sobel_y = np.array([[[1.0, 0.0, -1.0],
                     [2.0, 0.0, -2.0],
                     [1.0, 0.0, -1.0]],
                    [[1.0, 0.0, -1.0],
                     [2.0, 0.0, -2.0],
                     [1.0, 0.0, -1.0]],
                    [[1.0, 0.0, -1.0],
                     [2.0, 0.0, -2.0],
                     [1.0, 0.0, -1.0]]])


def filtro_media_cero(img, k_x, k_y, rgb=False):
    img = img.astype('float') / 255.0
    if rgb:
        g_x = sc.convolve(img, k_x)
        g_y = sc.convolve(img, k_y)
    else:
        g_x = sc.convolve(img[:, :, 0], k_x[:, :, 0])
        g_y = sc.convolve(img[:, :, 0], k_y[:, :, 0])
    return ((((g_x ** 2) + (g_y ** 2)) ** 1/2))


plt.figure(1, figsize=(7, 7))

plt.subplot(211)
plt.imshow(filtro_media_cero(img, prewitt_x, prewitt_y), cmap='gray')

plt.subplot(212)
plt.imshow(filtro_media_cero(img, sobel_x, sobel_y), cmap='gray')

plt.show()
