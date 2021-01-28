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

canny_k = np.array([[[-1.0, -1.0, -1.0],
                     [-1.0, 8.0, -1.0],
                     [-1.0, -1.0, -1.0]],
                    [[-1.0, -1.0, -1.0],
                     [-1.0, 8.0, -1.0],
                     [-1.0, -1.0, -1.0]],
                    [[-1.0, -1.0, -1.0],
                     [-1.0, 8.0, -1.0],
                     [-1.0, -1.0, -1.0]],
                    [[-1.0, -1.0, -1.0],
                     [-1.0, 8.0, -1.0],
                     [-1.0, -1.0, -1.0]]])


def filtro_canny(img, canny_k, x=8.0):
    img = img.astype('float') / 255.0
    canny_k[0, 1, 1] = float(x)
    return sc.convolve(img[:, :, 0], canny_k[0, :, :])


def filtro_media_cero(img, k_x, k_y):

    # Normalizamos la imagen a [0.0, 1.0]
    img = img.astype('float') / 255.0

    # Convolucionamos el kernel k_x y k_y con la iamgen
    g_x = sc.convolve(img[:, :, 0], k_x[0, :, :])
    g_y = sc.convolve(img[:, :, 0], k_y[0, :, :])

    # Devolvemos la combinacion de g_x y g_y -> sqrt(gx^2 + gy^2)
    return ((((g_x ** 2) + (g_y ** 2)) ** 1/2))


# Mostrar resultados filtrado
plt.figure(1, figsize=(4, 7))

plt.subplot(311)
plt.title('PREWITT')
plt.imshow(filtro_media_cero(img, prewitt_x, prewitt_y), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(312)
plt.title('SOBEL')
plt.imshow(filtro_media_cero(img, sobel_x, sobel_y), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(313)
plt.title('CANNY')
plt.imshow(filtro_canny(img, canny_k), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
