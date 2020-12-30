from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/vignetting.jpg')
height, width, rgb = img.shape
original = img.copy()

# Generamos una máscara de vignetting a partir de una distribución Gaussiana.
kernel_w = cv2.getGaussianKernel(width, 300)
kernel_h = cv2.getGaussianKernel(height, 300)
kernel = kernel_h * kernel_w.T

# Normalizamos la máscara para que tome valores en el intervalo [0,1]
mask = kernel / np.max(kernel)

redi = original[:, :, 0]
greeni = original[:, :, 1]
bluei = original[:, :, 2]

for c in range(rgb):
    img[:, :, c] = np.where((img[:, :, c] / mask) <
                            255, img[:, :, c] / mask, 255)


redf = img[:, :, 0]
greenf = img[:, :, 1]
bluef = img[:, :, 2]

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    nrows=2, ncols=3, figsize=(20, 5))
ax1.set_title('Histograma del color rojo en la imagen original')
ax1.hist(redi.ravel(), 256, [0, 256], color='r')
ax2.set_title('Histograma del color verde en la imagen original')
ax2.hist(greeni.ravel(), 256, [0, 256], color='g')
ax3.set_title('Histograma del color azul en la imagen original')
ax3.hist(bluei.ravel(), 256, [0, 256], color='b')
ax4.set_title('Histograma del color rojo en la imagen final')
ax4.hist(redf.ravel(), 256, [0, 256], color='r')
ax5.set_title('Histograma del color verde en la imagen final')
ax5.hist(greenf.ravel(), 256, [0, 256], color='g')
ax6.set_title('Histograma del color azul en la imagen final')
ax6.hist(bluef.ravel(), 256, [0, 256], color='b')
plt.tight_layout()
plt.show()

# cv2.imshow()
cv2_imshow(original)
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
