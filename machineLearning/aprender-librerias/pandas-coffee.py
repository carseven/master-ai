# %%
import pandas as pd
import os

# %%
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(THIS_FOLDER + '/coffees.csv')

data.head()

# Localiza una columna
data.loc[2]

# Como una lista, se indexa por un entero.
data.iloc[2:10]

# Cada columna del dataframe es una serie. Se puede indexar.
data.coffees[:10]
data["coffees"][:10]

len(data)
data.describe()

# %% Buscar los valores nulos
data.coffees.isnull()
data[data.coffees.isnull()]

# %% Convertir la columna coffees a float

"""Todos los datos que dan error al convertirse, se convertiran en NaN
 NaN no da error al covertir en float, se considera un numero valido."""
data.coffees = pd.to_numeric(data.coffees, errors='coerce')

"""Los nulos se pueden borrar directamente o se pueden interpolar.
Son lo mismo, data.dropna() devuelve el objeto pero lo tenemos que guardar.
data.dropna(inplace=True) otra forma de hacerlo"""
data = data.dropna()


"""Lo hacemos ahora xq no se puede convertir a int nulos."""
data.coffees = data.coffees.astype(int)

data.timestamp = pd.to_datetime(data.timestamp)
data.dtypes
