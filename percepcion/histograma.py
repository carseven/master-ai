import os
import numpy as np
import matplotlib.pyplot as plt


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
ejemplo = plt.imread(THIS_FOLDER + "/img/fruta.jpg")


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
    plt.figure(1, figsize=(7, 7))
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
    plt.show()

    return lista


lista_hist = histograma(ejemplo)
