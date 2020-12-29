# %% Paquetes necesarios
import numpy as np
import matplotlib.pyplot as plt
import os

# %% Cargamos imagen a procesar
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
# Es necesario convertir los pixeles a un rango de 0. y 1. para evitar
# overflow en la exponencial.
infraex = plt.imread(
    THIS_FOLDER + "/img/FOTO-OSCURA.jpg").astype('float') / 255.0
sobreex = plt.imread(THIS_FOLDER + "/img/fruta.jpg").astype('float') / 255.0
# %%


def compare_image_rgb(img_original, img_modify):
    plt.figure(1, figsize=(7, 7))

    plt.subplot(211)
    plt.imshow(img_original)

    plt.subplot(212)
    plt.imshow(img_modify)

    plt.show()

# %% Algotimo imagenes RGB mediante bucles


def autoConstrasteRGB(img, alpha, valor_max_entrada, valor_max_salida):
    const = valor_max_entrada / \
        (np.log(1 + np.exp(alpha - 1)) * valor_max_salida)

    salida = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for channel in range(img.shape[2]):
                salida[x][y][channel] = const * \
                    np.log(1 + (np.exp(alpha) - 1) * img[x, y, channel])
    return salida

# %% Algoritmo en RGB (Vecotrizado)


def trans_logaritmica(img, alpha, max_entrada, max_salida):
    const = max_salida / (np.log(1 + np.exp(alpha - 1)) * max_entrada)
    return const * np.log(1 + (np.exp(1) - 1) * img)

# %%


def trans_exponencial(img, alpha, max_entrada, max_salida):
    const = max_salida / (np.power(1 + alpha, max_entrada) - 1)
    return const * (np.power(1 + alpha, img) - 1)


# %%
compare_image_rgb(infraex, trans_logaritmica(infraex, 1, 1, 1))

compare_image_rgb(sobreex, trans_exponencial(sobreex, 10, 1, 1))
