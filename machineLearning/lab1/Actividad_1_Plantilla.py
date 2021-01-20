# This Python file uses the following encoding: utf-8

##############################
#UNIR
#
#Laboratorio. Arboles y random forest para regresión y clasificación
#Nombre del alumno: 
####################################
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession
#####################################################################
#Leer los datos los datos del USA Housing Dataset:
spark = SparkSession\
	.builder\
	.appName("RandomForestClassifierExample")\
	.getOrCreate()

USA_Housing = spark.read.format("libsvm").load("housing_train.csv")
print(USA_Housing)

#####################################################################
#1. An�lisis descriptivo de los datos:
#############################################################
  # De las variables num�ricas hallar datos estad�sticos: m�ximo, m�nimo, media, mediana y cuartiles



#############################################################

#############################################################
#De las variables categ�ricas, listar las diferentes categor�as 
#y hallar la frecuencia de cada una de ellas. 


############################################################

############################################################
#Crear la matriz de correlaciones con las columnas num�ricas


#Encontrar las correlaciones m�s altas, que superen el 0.75

#Revisar las correlaciones m�s altas y decidir si eliminar o no la columna


############################################################
#2. Tratamiento de missing. Si existen valores faltantes, 
#decidir si eliminar los registros, llenarlos con valores como
#la media, la mediana o la moda y justifique su respuesta
############################################################
#Identificar variables con valores faltantes

# Identificar columnas con valores faltantes superiores al 10% de los datos y eliminar columnas

# Identificamos NAs de variables que no superan el 10%

#Analizar las variables que tengan informaci�n sobre el garage, reemplazar valores NA con una categor�a adicional denominada "None"

#Analizar la variable MasVnrArea y decidir cual es la mejor manera de llenar los valores NA

#Elegir dos variables categ�ricas diferentes a las relacionadas con garage, analizar la mejor forma de reemplazar valores nulos. 
#En las dem�s variables categ�ricas los valores nulos los podemos reemplazar con la moda y las variables num�ricas con cero


# Verificar que no existen valores NA en el conjunto de datos

##########################################################

					  #3. Problema de Regresi�n
#########################################
#Arbol de decisi�n para problema de regresi�n

#Separar la data en entrenamiento y test

#Crear el �rbol

#Analizar si podar el �rbol o no. 

#Graficar el �rbol


#Predecir los valores de test 

#Hallar m�tricas de evaluaci�n del algoritmo, puede elegir entre el error cuadr�tico medio o su ra�z, puede ser otro de su elecci�n

#Comentar los resultados



###########################################################################
# Random Forest problema de regresi�n

#Datos de entrenamiento y datos de test

#Crear el modelo


#Predecir los valores de test 

#Hallar m�tricas de evaluaci�n del algoritmo, puede elegir entre el error cuadr�tico medio o su ra�z, puede ser otro de su elecci�n

#Comentar los resultados

######################################################################
#Comentar los resultados de los dos clasificadores.

##################################################################################


####################################################################################
#4. Problema de clasificaci�n

###################################################################################
#Preparar las categorias para la variable "y"

#grupo1 SalePrice menor o igual a 100 000, 
#grupo2 SalePrice entre 100 001 y 500 000,  
#grupo3 SalePrice mayor o igual a 500 001.

#crear la nueva columna

#eliminar la columna SalesPrice para evitar problemas de sobre ajuste en el modelo
#Arbol de decisi�n para problema de regresi�n

#Separar datos de entrenamiento y test

##################################################################################
#�rboles de decisi�n para el problema de clasificaci�n


#Crear el �rbol

#Analizar si podar el �rbol o no. 

#Graficar el �rbol


#Predecir los valores de test 

#Crear matriz de confusi�n y hallar exactitud

#Comentar los resultados

#################################################################################


#######################################################################################3
#Random Forest para el problema de clasificaci�n


#Crear el modelo


#Predecir los valores de test 

#Crear matriz de confusi�n y hallar exactitud

#Comentar los resultados

######################################################################
#Comentar los resultados de los dos clasificadores.

##################################################################################


