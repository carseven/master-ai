##############################
#UNIR
#
#Actividad. SVM y redes neuronales
#Nombre del alumno: 
####################################

library(neuralnet)
library(CatEncoders)
library(caret)
#Leer los datos los datos 
mobileData <- read.csv("datasets_train.csv")
#####################################################################
#1. Análisis descriptivo de los datos:
#############################################################
#Pasar a factor las columnas categóricas

# De las variables numéricas hallar datos estadísticos: máximo, mínimo, media, mediana y cuartiles



#############################################################

#############################################################
#De las variables categóricas, listar las diferentes categorías 
#y hallar la frecuencia de cada una de ellas. 


############################################################

############################################################
#Crear la matriz de correlaciones con las columnas numéricas


#Encontrar las correlaciones más altas, que superen el 0.75

#Revisar las correlaciones más altas y decidir si eliminar o no la columna




#####################################################################
#2. Aplicar SVM
#############################################################
#Separar la data en entrenamiento y test

# Crear un primer modelo


#Aplicar algoritmo de optimización al modelo e imprimir el mejor modelo

#Predecir clasificación con el conjunto de datos de test

#Crear matriz de confusión y hallar exactitud

#####################################################################
#3. Aplicar Redes neuronales
#############################################################
#Reescalar datos 

#OneHotEncoderData

#Separar la data en entrenamiento y test

#Crear modelo de red neuronal

#Dibujar modelo

#Predecir clasificación en conjunto de test

#Crear matriz de confusión y hallar exactitud

#####################################################################
#4. Comentarios sobre los resultados
#############################################################
