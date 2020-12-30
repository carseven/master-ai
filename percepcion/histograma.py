import os
import numpy as np
import matplotlib.pyplot as plt


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
ejemplo = plt.imread(THIS_FOLDER + "/img/vignetting.jpg")


def histograma(img):
    """Calcula y representa el histograma de una imagen RGB.

    Args:
        img (Numpy array 3d): Un imagen RGB

    Returns:
        List: Devuelve una lista con el array del histograma de cada canal RGB.
        Cada canal contendra un array de dos dimensiones, la primera contendra
        las intensidades y la segunda
    """

    # Configuracion general de la grafica
    plt.figure(4, figsize=(7, 7))
    plt.suptitle('Histograma RGB', fontsize=30)

    # Lista donde almacenaremos histograma de cada canal
    lista = []

    # Calculamos y mostramos los 3 canales RGB de la imagen entrada.
    for c in range(img.shape[2]):

        # Calcular histograma
        vector = img[:, :, c].flatten()
        unique, counts = np.unique(vector, return_counts=True)
        lista.append(np.asarray((unique, counts)))

        # Configuracion subplot de cada canal
        plt.subplot(3, 1, c + 1)
        plt.xlim(0, 255)
        plt.ylim(0, np.max(lista[c][1]) + 50)

        # Configuracion subplot especifica de cada canal
        if c == 0:
            plt.plot(lista[c][0], lista[c][1], color='r')
            plt.title('Canal Rojo')
        elif c == 1:
            plt.plot(lista[c][0], lista[c][1], color='g')
            plt.title('Canal Verde')
            plt.ylabel('Nº pixeles')

        elif c == 2:
            plt.plot(lista[c][0], lista[c][1], color='b')
            plt.title('Canal Azul')
            plt.xlabel('Intensidad')

    plt.tight_layout()

    return lista


def histograma_acumulado(img, bins):
    """Realiza una ecualización de la imagén de entrada

    Args:
        img (numpy array): Array con la imagén de entrada.
        bins (int): Número de intensidades de la imagén de entrada.

    Returns:
        Numpy array: Devuelve el array con la imagen de entrada tras aplicar la
        ecualización.
    """
    flat = img.flatten()

    plt.figure(1)
    plt.hist(flat, bins=50)

    # Calcular histograma
    histogram = np.zeros(bins)
    for pixel in flat:
        histogram[pixel] += 1

    # Histograma acumulado
    histogram = iter(histogram)
    histogram_acum = [next(histogram)]
    for i in histogram:
        histogram_acum.append(histogram_acum[-1] + i)

    histogram_acum = np.array(histogram_acum)

    # Normalizamos a 0-255 el histograma acumulado
    nj = (histogram_acum - histogram_acum.min()) * 255
    N = histogram_acum.max() - histogram_acum.min()
    histogram_acum = (nj / N).astype('uint8')

    plt.figure(2)
    plt.plot(histogram_acum)

    # Obtener los valores del histograma acumulado para cada indice de la image
    img_final = histogram_acum[flat]

    # Forma original de la imagen
    img_final = np.reshape(img_final, img.shape)

    return img_final


img_final = histograma_acumulado(ejemplo, 255)
plt.figure(3)
plt.imshow(img_final)


histograma(ejemplo)
plt.figure(5)
plt.imshow(ejemplo)
plt.show()
