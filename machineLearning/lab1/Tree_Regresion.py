import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

data = pd.read_csv('housing_train.csv')
data.info()

#Preparado de datos sin mucho sentimiento
numericDf = data.select_dtypes(include=numerics)
numericDf = numericDf.fillna(numericDf.mean())
numericDf.info()

# Árboles de decisión
# Separamos el conjunto de test y train --> Intentaremos predecir el campo OverallQual
x_train,x_test,y_train,y_test = train_test_split(numericDf, numericDf.OverallQual, test_size = 0.30, random_state = 42)

# Finding best parameters for decision tree
dt = DecisionTreeRegressor(random_state=0)

dt_params = {'max_depth':np.arange(1,50,2),'min_samples_leaf':np.arange(2,30)}

gs_dt = GridSearchCV(dt,dt_params,cv=7)#7 folds
gs_dt.fit(x_train,y_train)
bestParams = gs_dt.best_params_

print("Grid Search:")
print(gs_dt)
print("Best params:")
print(bestParams)

# Entrenamos con los parámetros
# max_deph = 5, leaf = 3 (recomendados en bestParams)
# RMLSE for the data: 0.01698225529205052
# MAE: 0.0030441400304414006
# MSE: 0.002029426686960934
dtr=DecisionTreeRegressor(max_depth=5,min_samples_leaf=3)
model = dtr.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error, mean_squared_error

msle=mean_squared_log_error(y_pred,y_test)
rmsle=np.sqrt(msle)
print('RMLSE for the data:',rmsle) # For decision tree

print('MAE:',mean_absolute_error(y_pred,y_test))
print('MSE:',mean_squared_error(y_pred,y_test))

from sklearn import tree
r_texto = tree.export_text(dtr)

print(r_texto)
