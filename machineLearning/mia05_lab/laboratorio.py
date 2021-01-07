##############################
# %% UNIR
#
# Laboratorio. Árboles y random forest para regresión y clasificación
# Nombre del alumno:
####################################

import pandas as pd

# %% Leer los datos los datos del USA Housing Dataset:
housing_train = pd.read_csv('housing_train.csv')

# %% Analisis exploratorio
print('Dimensiones:', housing_train.shape)
print(list(housing_train.columns))
print(housing_train.info())
print(housing_train.describe())

# %%
