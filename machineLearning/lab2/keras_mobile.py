# Calculo y manipulación de datos
import pandas as pd
import numpy as np
from numpy.lib.function_base import average

# Modelizacion
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn import svm
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras import models
from keras import layers

# Utilidades
import os

# Visualizacion
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# Descargar dataset
!rm / content/datasets_train.csv
!wget https: // raw.githubusercontent.com/carseven/master-ai/main/machineLearning/lab2/datasets_train.csv

# Referencias
# https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/
# https://stackabuse.com/deep-learning-in-keras-data-preprocessing/
# https://www.kaggle.com/sanchit2843/neuralnetworkkeras
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/

# Que pandas muestre todas las columnas
pd.set_option("display.max_rows", 30, "display.max_columns", None)
pylab.rcParams['figure.figsize'] = (16.0, 7.0)

"""## Cargar dataset"""

data = pd.read_csv('/content/datasets_train.csv')

"""## Análisis descriptivo de los datos

### Analisis inicial y distinción de los tipos de variables
"""

data.head(5)

data.info()

"""Explicar un poco el dataset y la caracteristicas de cada columna.
battery_power: Total energy a battery can store in one time measured in mAh
blue: Has bluetooth or not
clock_speed: speed at which microprocessor executes instructions
dual_sim: Has dual sim support or not
fc: front camera mega pixeles
four_g: has 4g or not
int_memory: Internal Memory in Gigabytes
m_dep: Mobile Depth in cm
mobile_wt: Weight of mobile phone
n_cores: Number of cores of processor
pc: Primary Camera mega pixels
px_height: Pixel Resolution Height
px_width: Pixel Resolution Width
ram: Random Access Memory in Megabytes
sc_h: Screen Height of mobile in cm
sc_w: Screen Width of mobile in cm
talk_time: longest time that a single battery charge will last when you are
three_g: Has 3G or not
touch_screen: Has touch screen or not
wifi: Has wifi or not
Columnas a predecir en la clasificacion.
price_range: This is the target variable with value of
0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

Distingir variables númericas de las categoricas
"""

cat_columns = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen',
               'wifi', 'price_range']

num_columns = list(set(data.columns) - set(cat_columns))

print('Variables númericas:')
print(num_columns)
print('\n')
print('Variables categoricas:')
print(cat_columns)

"""### Datos estadísticos de las variables númericas"""

print('Datos estadíticos de las variables númericas:')
for column in num_columns:
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

"""### Frecuencia de las variables categoricas"""

for column in cat_columns:
    print(f'Variable {column}:')
    print(f'Listado categorias: \n{data[column].unique()}')
    print(f'Frecuencia categorias: \n{data[column].value_counts()}')
    print('\n')

"""### Distribución de las variables númericas"""

fig, axs = plt.subplots(len(num_columns), 1, figsize=(15, 50))

count = 0
for column in num_columns:
    fp = sb.distplot(data[column], ax=axs[count])
    fp.set_title(f'Distribución de {column}')
    count += 1

plt.tight_layout()

"""### Distribución de las variables categoricas"""

fig, axs = plt.subplots(len(cat_columns), 1, figsize=(15, 50))

count = 0
for column in cat_columns:
    fp = sb.countplot(data[column], ax=axs[count])
    fp.set_title(f'Distribución de {column}')
    fp.set_xticklabels(data[column].unique())
    count += 1

"""### Matriz de correlación"""

# Calcular correlacion de las variables numericas.
df_matrix_corr = data[num_columns].corr(method='pearson').abs()

# Triangulo inferior de la matrix de correlacion
mask = np.triu(np.ones(df_matrix_corr.shape)).astype(np.bool)

# Grafico de la matriz de correlacion
f, ax = plt.subplots(figsize=(15, 20))
ax.set_yticklabels(df_matrix_corr.columns[1:], rotation=0)
ax.set_xticklabels(df_matrix_corr.columns[:-1], rotation=45)
heatmap = sb.heatmap(
    df_matrix_corr,
    mask=mask,
    square=True,
    linewidths=1,
    cmap=sb.light_palette('seagreen', as_cmap=True),
    cbar_kws={'shrink': .4, 'ticks': [-1, -.5, 0, 0.5, 1]},
    vmin=0,
    vmax=1,
    annot=True,
    annot_kws={"size": 10},
    fmt=".2f"
)

umbral = 0.75
corr_stack = df_matrix_corr.where(mask).stack().reset_index()
corr_stack.columns = ['Row', 'Column', 'Correlation Value']
corr_75 = corr_stack[(corr_stack['Correlation Value'] > umbral) &
                     (corr_stack['Correlation Value'] < 1)]
corr_75 = corr_75.sort_values(
    by='Correlation Value',
    kind="quicksort",
    ascending=False).drop_duplicates(keep='first')
print(corr_75)

"""Se aprecia claramente una relación en las variables como pc, fc. sc_w y sc_h,
px_width y px_px_height
Mostrar distribuciones, comparar las distribuciones con estas.
No se ha encontrado ninguna correlación que supere el 75%

### Tratamiento de missing
"""

percent_null = (data.isnull().sum() / len(data)) * 100
print(percent_null)

"""## Máquina de Vectores de Soporte

### Pre-procesado
"""

x = data.drop(columns='price_range')
y = data['price_range']

"""Hold out"""

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print("Datos de entrenamiento:")
print(f"Samples: {x_train.shape}")
print(f"Labels: {y_train.shape}\n")
print("Datos de test:")
print(f"Samples: {x_test.shape}")
print(f"Labels: {y_test.shape}\n")

"""## Modelo"""

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(svm.SVC(), parameters, scoring='accuracy', cv=5)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)

svm_model = svm.SVC(kernel=clf.best_params_[
                    'kernel'], C=clf.best_params_['C'], random_state=42)
svm_model = svm_model.fit(x_train, y_train)
predicted = svm_model.predict(x_test)
print(predicted)


def counts_from_confusion(confusion_matrix):
    count = []

    # Iterate through classes and store the counts
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]

        fn_mask = np.zeros(confusion_matrix.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(confusion_matrix, fn_mask))

        fp_mask = np.zeros(confusion_matrix.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion_matrix, fp_mask))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(confusion_matrix, tn_mask))

        count.append({'Clase': i,
                      'TP': tp,
                      'FN': fn,
                      'FP': fp,
                      'TN': tn})

    tp = (count[0]['TP'] + count[1]['TP'] + count[2]
          ['TP'] + count[3]['TP'])
    tn = (count[0]['TN'] + count[1]['TN'] + count[2]
          ['TN'] + count[3]['TN'])
    fp = (count[0]['FP'] + count[1]['FP'] + count[2]
          ['FP'] + count[3]['FP'])
    fn = (count[0]['FN'] + count[1]['FN'] + count[2]
          ['FN'] + count[3]['FN'])

    return tp, tn, fp, fn, count


confusion_matrix = metrics.confusion_matrix(y_test, predicted)
tp, tn, fp, fn, dict_confusion = counts_from_confusion(confusion_matrix)

accuracy = metrics.accuracy_score(y_test, predicted)
recall = metrics.recall_score(y_test, predicted, average='micro')
f1 = metrics.f1_score(y_test, predicted, average='micro')
specificity = tn/(tn + fp)
sensibility = tp/(tp + fn)


print("Métricas evaluación:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Sensibility: {sensibility:.2f}")

# Visualizar matriz de confusión
metrics.plot_confusion_matrix(svm_model, x_test, y_test)
plt.show()
print(dict_confusion)

"""## Redes neuronales

### Pre-procesado
"""

data = pd.read_csv('/content/datasets_train.csv').to_numpy()
x = data[:, :-1]
y = data[:, -1].astype('int')

# Normalización: Media y desviación.
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)

# One-hot enconding
y = to_categorical(y)
print(y)

"""Hold out"""

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=42)

print(f"Datos entrenamiento y validacion: {x_train.shape} {y_train.shape}")
print(f"Datos test: {x_test.shape} {y_test.shape}")

kfold = KFold(n_splits=10, shuffle=True)

fold = 0
accuracy_fold = []
loss_fold = []
for train, validate in kfold.split(x_train, y_train):
    # Datos entrenamiento
    x_train_fold = x_train[train]
    y_train_fold = y_train[train]

    # Datos validacion
    x_valdiate_fold = x_train[validate]
    y_valdiate_fold = y_train[validate]

    # Model
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(x.shape[1],)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Entrenamiento
    history = model.fit(x_train_fold, y_train_fold,
                        validation_data=(x_valdiate_fold, y_valdiate_fold),
                        batch_size=16,
                        epochs=50,
                        verbose=0)

    # Test
    prediction_fold = model.evaluate(x_test, y_test, verbose=0)
    y_predicted = model.predict(x_test)

    if fold == 0:
        model.summary()

    print("----------------------------------------------------------------")
    print(f"Entrenando para el Fold {fold}")
    print(f"{model.metrics_names[0]} of {prediction_fold[0]}")
    print(f"{model.metrics_names[1]} of {prediction_fold[1]*100}%")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    mc = metrics.confusion_matrix(y_test.argmax(axis=1),
                                  y_predicted.argmax(axis=1))
    fig, ax = plt.subplots()
    ax = sb.heatmap(
        mc,
        fmt='.4g',
        annot=True,
        annot_kws={'size': 16},
        cmap='Blues_r',
        cbar=False)

    plt.title('Matriz de confusión')
    plt.ylabel('Clase actual')
    plt.xlabel('Clase de predicción')
    plt.ylim(4, 0)
    plt.show()

    accuracy_fold.append(prediction_fold[1] * 100)
    loss_fold.append(prediction_fold[0])

    fold += 1

acc_mean = np.array(accuracy_fold).mean()
loss_mean = np.array(loss_fold).mean()
print(f"\nLa accuracy media es de {acc_mean}")
print(f"La loss media es de {loss_mean}")

predicted = model.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1),
                                            predicted.argmax(axis=1))
tp, tn, fp, fn, dict_confusion = counts_from_confusion(confusion_matrix)

accuracy = metrics.accuracy_score(y_test.argmax(axis=1),
                                  predicted.argmax(axis=1))
recall = metrics.recall_score(y_test.argmax(axis=1),
                              predicted.argmax(axis=1), average='micro')
f1 = metrics.f1_score(y_test.argmax(axis=1),
                      predicted.argmax(axis=1), average='micro')
specificity = tn/(tn + fp)
sensibility = tp/(tp + fn)


print("Métricas evaluación:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Sensibility: {sensibility:.2f}")


mc = metrics.confusion_matrix(y_test.argmax(axis=1),
                              y_predicted.argmax(axis=1))
fig, ax = plt.subplots()
ax = sb.heatmap(
    mc,
    fmt='.4g',
    annot=True,
    annot_kws={'size': 16},
    cmap='Blues_r',
    cbar=False)

plt.title('Matriz de confusión')
plt.ylabel('Clase actual')
plt.xlabel('Clase de predicción')
plt.ylim(4, 0)
plt.show()
