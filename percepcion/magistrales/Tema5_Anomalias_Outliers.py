#%% importar
import numpy as np
import matplotlib.pyplot as plt
import math

#%% 
def outlinerRemoval(s, t):
    # s: Dataset con las muestras de la variable aleatoria
    # t: Porcentaje umbral por el cual consideramos que las muetras son anomalías


    # Calculo percentil inferior y superior
    s_min = np.percentile(s, 100 - t) 
    s_max = np.percentile(s, t)

    return [s_min, s_max]

# %%
# Probabilidad de ocurrencia para considerarse outlier
probabilidad = 99.8

x = np.random.normal(loc=0, scale=0.2, size=1000)

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(x)
ax1.set_title('Samples')
ax2.hist(x, bins='fd')
ax2.set_title('Histogram')
plt.show()

limites = outlinerRemoval(x, probabilidad)

print('Number of samples ' + str(len(x)) + 
    '\nUmbral inferrior: ' + str(limites[0]) +
    '\nUmbral inferrior: ' + str(limites[1]))
# Al ser una distribución gaussiana, el umbral inferior es el mismo que el
# superior, como se observa en el histpgrama.
# %%
