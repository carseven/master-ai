##############################
#UNIR
#
#Laboratorio. √Årboles y random forest para regresi√≥n y clasificaci√≥n
#Nombre del alumno: 
####################################

#####################################################################
#Leer los datos los datos del USA Housing Dataset:
USA_Housing<- read.csv("housing_train.csv")
str(USA_Housing)

#####################################################################
#1. An·lisis descriptivo de los datos:
#############################################################
  # De las variables numÈricas hallar datos estadÌsticos: m·ximo, mÌnimo, media, mediana y cuartiles



#############################################################

#############################################################
#De las variables categÛricas, listar las diferentes categorÌas 
#y hallar la frecuencia de cada una de ellas. 


############################################################

############################################################
#Crear la matriz de correlaciones con las columnas numÈricas


#Encontrar las correlaciones m·s altas, que superen el 0.75

#Revisar las correlaciones m·s altas y decidir si eliminar o no la columna


############################################################
#2. Tratamiento de missing. Si existen valores faltantes, 
#decidir si eliminar los registros, llenarlos con valores como
#la media, la mediana o la moda y justifique su respuesta
############################################################
#Identificar variables con valores faltantes

# Identificar columnas con valores faltantes superiores al 10% de los datos y eliminar columnas

# Identificamos NAs de variables que no superan el 10%

#Analizar las variables que tengan informaciÛn sobre el garage, reemplazar valores NA con una categorÌa adicional denominada "None"

#Analizar la variable MasVnrArea y decidir cual es la mejor manera de llenar los valores NA

#Elegir dos variables categÛricas diferentes a las relacionadas con garage, analizar la mejor forma de reemplazar valores nulos. 
#En las dem·s variables categÛricas los valores nulos los podemos reemplazar con la moda y las variables numÈricas con cero


# Verificar que no existen valores NA en el conjunto de datos

##########################################################

                      #3. Problema de RegresiÛn
#########################################
#Arbol de decisiÛn para problema de regresiÛn

#Separar la data en entrenamiento y test

#Crear el ·rbol

#Analizar si podar el ·rbol o no. 

#Graficar el ·rbol


#Predecir los valores de test 

#Hallar mÈtricas de evaluaciÛn del algoritmo, puede elegir entre el error cuadr·tico medio o su raÌz, puede ser otro de su elecciÛn

#Comentar los resultados



###########################################################################
# Random Forest problema de regresiÛn

library(randomForest)
library(caret)

#Datos de entrenamiento y datos de test

#Crear el modelo


#Predecir los valores de test 

#Hallar mÈtricas de evaluaciÛn del algoritmo, puede elegir entre el error cuadr·tico medio o su raÌz, puede ser otro de su elecciÛn

#Comentar los resultados

######################################################################
#Comentar los resultados de los dos clasificadores.

##################################################################################


####################################################################################
#4. Problema de clasificaciÛn

###################################################################################
#Preparar las categorias para la variable "y"

#grupo1 SalePrice menor o igual a 100 000, 
#grupo2 SalePrice entre 100 001 y 500 000,  
#grupo3 SalePrice mayor o igual a 500 001.

#crear la nueva columna

#eliminar la columna SalesPrice para evitar problemas de sobre ajuste en el modelo
#Arbol de decisiÛn para problema de regresiÛn

#Separar datos de entrenamiento y test
##################################################################################

#¡rboles de decisiÛn para el problema de clasificaciÛn

#Crear el ·rbol

#Analizar si podar el ·rbol o no. 

#Graficar el ·rbol


#Predecir los valores de test 

#Crear matriz de confusiÛn y hallar exactitud

#Comentar los resultados

#################################################################################


#######################################################################################3
#Random Forest para el problema de clasificaciÛn


#Crear el modelo


#Predecir los valores de test 

#Crear matriz de confusiÛn y hallar exactitud

#Comentar los resultados

######################################################################
#Comentar los resultados de los dos clasificadores.

##################################################################################


