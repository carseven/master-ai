# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as pylab
import numpy as np
import seaborn as sns
import os

pylab.rcParams['figure.figsize'] = (16.0, 7.0)
# %% Leer los datos los datos del USA Housing Dataset:
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(THIS_FOLDER + '/housing_train.csv')

# %% Analisis exploratorio

# Discriminar en variables numericas y categoricas
variables_total = data.columns
variables_num = list(data._get_numeric_data().columns)
variables_cat = list(set(variables_total) - set(variables_num))

# Convertimos la columnas categoricas en objetos categoricos
# for column in variables_cat:
#     data[column] = pd.Categorical(data[column])

print(data.info())
print(variables_num)
print(variables_cat)
print(len(variables_num) + len(variables_cat))
print(len(variables_total))
# %% De las variables numericas: maximo, minimo, media, mediana y cuartiles.

print('Datos estadíticos de las variables númericas:')
for column in variables_num:
    cuartiles = data[column].quantile([.25, .50, .75])

    print('Variable ' + column + ':')
    print('MAX:', data[column].max())
    print('MIN:', data[column].min())
    print('MEDIA:', data[column].mean())
    print('MEDIANA:', data[column].median())
    print('25%:', cuartiles.iloc[0])
    print('50%:', cuartiles.iloc[1])
    print('75%:', cuartiles.iloc[2])
    print('\n')

# %% De las variables categoricas: listado categorias y frecuencia aparicion

for column in variables_cat:
    print(f'Variable {column}:')
    print(f'Listado categorias: \n{data[column].unique()}')
    print(f'Frecuencia categorias: \n{data[column].value_counts()}')
    print('\n')

# %% Matriz de correlacion (Forma 1)

df_matrix_corr = data.corr().abs()

# Graficar la matriz de correlacion
f = plt.figure(figsize=(19, 15))
plt.matshow(df_matrix_corr, fignum=f.number)
plt.xticks(range(data.select_dtypes(['number']).shape[1]),
           data.select_dtypes(['number']).columns,
           fontsize=14,
           rotation=90)
plt.yticks(range(data.select_dtypes(['number']).shape[1]),
           data.select_dtypes(['number']).columns,
           fontsize=14)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
# %% Matriz de correlacion (Forma 2)

df_matrix_corr = data.corr().abs()
mask = np.zeros_like(df_matrix_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(30, 30))
heatmap = sns.heatmap(
    df_matrix_corr,
    mask=mask,
    square=True,
    linewidths=.2,
    cmap='coolwarm',
    cbar_kws={'shrink': .4, 'ticks': [-1, -.5, 0, 0.5, 1]},
    vmin=0,
    vmax=1,
    annot=True,
    annot_kws={"size": 8}
)
ax.set_yticklabels(df_matrix_corr.columns, rotation=0)
ax.set_xticklabels(df_matrix_corr.columns, rotation=45)
# %% Eliminar las columnas que superan 0.75 (Forma 1)


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    print(colname)
                    # del dataset[colname] # deleting the column from the dataset


correlation(data, 0.75)

# %% Eliminar las columnas que superan 0.75 (Forma 2)

# Obtenemos el triangulo superior de la matriz de correlacion
upper = df_matrix_corr.where(
    np.triu(np.ones(df_matrix_corr.shape), k=1).astype(np.bool)
)
to_drop = [
    column for column in upper.columns if any(upper[column] > 0.75)]

# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# %% Buscar variables que superen el 10% de nulos.

percent_null = (data.isnull().sum() / len(data)) * 100

higher_10 = [
    column for column in percent_null.index if percent_null[column] >= 10.0]
print('Columnas igual o más de 10% nulos:')
print(higher_10)
print()
# Buscar variables que NAs no superen 10%.

lower_10_percent = [
    column for column in percent_null.index if percent_null[column] < 10.0 and percent_null[column] > 0.0]

print('Columnas menor 10% nulos, pero con algun nulo:')
print(lower_10_percent)

# %% Variables relacionadas con garaje remplazar NaN con una categoria # Nose

# %% Analizar MasVnrArea y decidir como reemplazar los NaN

# %%Elegir dos variables categ�ricas diferentes a las relacionadas con garage, analizar la mejor forma de reemplazar valores nulos.
# En las dem�s variables categ�ricas los valores nulos los podemos #reemplazar con la moda y las variables num�ricas con cero

# %% Verificar que no existen valores NA en el conjunto de datos
