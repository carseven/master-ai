# %%
import pandas as pd
import numpy as np
# %%
# Son como los diccionarios pero solo de una dimensión
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'], dtype='float64')

# Con numpy arrays
series = pd.Series(np.array([1, 2, 3]), index=['a', 'b', 'c'], dtype='float64')

# Con diccionarios.
dic = {'a': 1, 'b': 2, 'c': 3}
series = pd.Series(dic, dtype='float64')

print(series)

# %%
# Se puede acceder mediante como listas o diccionarios
print(series[1:], series['a'])

print(series.values)
print(series.index)
# %%
# Como se basan en arrays de numpy podemos hacer las mismas operaciones
# Pero ojo tienen que tener los mismos indices
# Podemos hacer operaciones basicas de numpy

serie1 = series = pd.Series([1, 2, 3], index=['a', 'b', 'c'], dtype='float64')
serie2 = series = pd.Series([4, 5, 6], index=['a', 'b', 'c'], dtype='float64')
print(serie1 + serie2, "", serie2 / serie1)
print(np.sqrt(serie1))
# %%
# Dataframe son estructuras de 2 dimensiones
# Como si fuesen una serie de 2 dimensiones

# Mediante diccionarios
dict = {
    'columna1': pd.Series([1, 2, 3], index=['a', 'b', 'c'], dtype='float64'),
    'columna2': pd.Series([4, 5, 6], index=['a', 'b', 'c'], dtype='float64')
}

data_frame_dict = pd.DataFrame(dict)
# print(data_frame_dict)
# print(data_frame_dict['columna1'])

# Añadir y eliminar columnas

serie3 = pd.Series([7, 8, 9], index=['a', 'b', 'c'], dtype='float64')

data_frame_dict['columna3'] = serie3
print(data_frame_dict)

del data_frame_dict['columna3']
print(data_frame_dict)

# Obtener una fila por la etiqueta
print(data_frame_dict.loc['a'])

# %% Obtener una fila por posición
print(data_frame_dict.iloc[0])
print(data_frame_dict.iloc[0:2])

# %%
filtro = np.less_equal(['columna1'],
                       data_frame_dict['columna2'])
print(filtro)
print(data_frame_dict[filtro])

# %% Mediante listas
lst = [{'a': 1, 'b': 2, 'c': 3},
       {'a': 4, 'b': 5, 'c': 6}]

data_frame_lst = pd.DataFrame(lst)
print(data_frame_lst)
# %%
# Dataframes incompletos
# Mediante listas
lst_incompleta = [{'a': 1, 'b': 2, 'c': 3},
                  {'a': 4, 'b': 5, 'd': 6}]

data_frame_lst_incompleta = pd.DataFrame(lst_incompleta)
print(data_frame_lst_incompleta)

# %% Se puede hacer operaciones con los dataframe pero ojo xq si los tamaños no
# son iguales obtendremos NaN
data_frame_doble = data_frame_lst_incompleta * 2
print(data_frame_doble)

# %% Transpuesta de un dataframe
print(data_frame_doble.T)

# %% Ordenar por,
print(data_frame_doble.sort_values(by='a'))

data_frame_doble.describe()

# De cada propiedad cuantas tenemos
data_frame_dict.groupby(by='columna1').count()

data_frame_dict.to_csv('ejemplo.csv')
df_ejemplo = pd.read_csv('ejemplo.csv')

del df_ejemplo['Unnamed: 0']
df_ejemplo
