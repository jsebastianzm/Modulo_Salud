import a_funciones as a
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor ##Ensamble con bagging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
import keras.backend as K


df_resultado = pd.read_csv('datos_resultado.csv')


#Se separan las bases de datos en la variable a predecir y las variables independientes
y = df_resultado["TIEMPO ESTANCIA"]
x = df_resultado.loc[:, ~df_resultado.columns.isin(['TIEMPO ESTANCIA'])]


#Eliminar columnas que no son necesarias
eliminar = ['FECHA', 'NRODOC']
x.drop(eliminar, axis = 1, inplace=True)

columns_object = x.select_dtypes(include = ['object'])
columns_object.columns

# a.info_columns(columns_object)

x = pd.get_dummies(x, columns=columns_object.columns)


columns_num = x.select_dtypes(include = ['int64', 'float64'])
columns_num.info()
scaler = StandardScaler()
x[columns_num.columns] = scaler.fit_transform(x[columns_num.columns])


m_rf = RandomForestRegressor()
m_svm = LinearSVR()


modelos=list([m_rf, m_svm])

var_names = a.sel_variables1(modelos,x,y,threshold="1.7*mean")
var_names.shape

X2=x[var_names] ### matriz con variables seleccionadas
X2.info()

# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=1)

param_grid_rf = [{'n_estimators': [50, 500, 100], 'min_samples_split': [10, 20, 3], 
               'max_leaf_nodes':[20,50,100]}]

tun_rf_r2=RandomizedSearchCV(m_rf,param_distributions=param_grid_rf,n_iter=3,scoring="r2")
tun_rf_r2.fit(x_train,y_train)
y_pred = tun_rf_r2.predict(x_train)

#Exactitud de modelo
print("Accuracy (Train): "+ str(tun_rf_r2.score(x_train,y_train)*100) + "%")
#Metricas de desempeño de entrenamiento
print("MSE entrenamiento: {}".format(mean_squared_error(y_train, y_pred)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_train, y_pred)))
print("R2 entrenamiento: {}".format(r2_score(y_train, y_pred)))


#Predicciones sobre el conjunto de test
y_hat = tun_rf_r2.predict(x_test)
#Exactitud de modelo
print("Accuracy (Test): "+ str(tun_rf_r2.score(x_test,y_test)*100) + "%")
#Metricas de desempeño de validación
print("MSE entrenamiento: {}".format(mean_squared_error(y_test, y_hat)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_test, y_hat)))
print("R2 entrenamiento: {}".format(r2_score(y_test, y_hat)))


#---------------------------------------------------------------------

tun_rf_MSE=RandomizedSearchCV(m_rf,param_distributions=param_grid_rf,n_iter=3,scoring="neg_mean_squared_error")
tun_rf_MSE.fit(x_train,y_train)
y_pred = tun_rf_MSE.predict(x_train)



#Exactitud de modelo
print("Accuracy (Train): "+ str(tun_rf_MSE.score(x_train,y_train)*100) + "%")
#Metricas de desempeño de entrenamiento
print("MSE entrenamiento: {}".format(mean_squared_error(y_train, y_pred)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_train, y_pred)))
print("R2 entrenamiento: {}".format(r2_score(y_train, y_pred)))


#Predicciones sobre el conjunto de test
y_hat = tun_rf_MSE.predict(x_test)
#Exactitud de modelo
print("Accuracy (Test): "+ str(tun_rf_MSE.score(x_test,y_test)*100) + "%")
#Metricas de desempeño de validación
print("MSE entrenamiento: {}".format(mean_squared_error(y_test, y_hat)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_test, y_hat)))
print("R2 entrenamiento: {}".format(r2_score(y_test, y_hat)))


#---------------------------------------------------------------------------

tun_rf_MAE=RandomizedSearchCV(m_rf,param_distributions=param_grid_rf,n_iter=3,scoring="neg_mean_absolute_error")
tun_rf_MAE.fit(x_train,y_train)
y_pred = tun_rf_MAE.predict(x_train)



#Exactitud de modelo
print("Accuracy (Train): "+ str(tun_rf_MAE.score(x_train,y_train)*100) + "%")
#Metricas de desempeño de entrenamiento
print("MSE entrenamiento: {}".format(mean_squared_error(y_train, y_pred)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_train, y_pred)))
print("R2 entrenamiento: {}".format(r2_score(y_train, y_pred)))


#Predicciones sobre el conjunto de test
y_hat = tun_rf_MAE.predict(x_test)
#Exactitud de modelo
print("Accuracy (Test): "+ str(tun_rf_MAE.score(x_test,y_test)*100) + "%")
#Metricas de desempeño de validación
print("MSE entrenamiento: {}".format(mean_squared_error(y_test, y_hat)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_test, y_hat)))
print("R2 entrenamiento: {}".format(r2_score(y_test, y_hat)))


#-----------------------------------------------------------------------

param_grid_svm = [{'C': [1, 1.5, 2], 'epsilon': [0.1, 0.5, 0.7], 'loss': ["epsilon_insensitive"], 
               'max_iter':[20,50,100], 'random_state':[123]}]

tun_svm_r2 = RandomizedSearchCV(m_svm,param_distributions=param_grid_svm,n_iter=3,scoring="r2")
tun_svm_r2.fit(x_train,y_train)
y_pred = tun_svm_r2.predict(x_train)

#Exactitud de modelo
print("Accuracy (Train): "+ str(tun_svm_r2.score(x_train,y_train)*100) + "%")
#Metricas de desempeño de entrenamiento
print("MSE entrenamiento: {}".format(mean_squared_error(y_train, y_pred)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_train, y_pred)))
print("R2 entrenamiento: {}".format(r2_score(y_train, y_pred)))


#Predicciones sobre el conjunto de test
y_hat = tun_svm_r2.predict(x_test)
#Exactitud de modelo
print("Accuracy (Test): "+ str(tun_svm_r2.score(x_test,y_test)*100) + "%")
#Metricas de desempeño de validación
print("MSE entrenamiento: {}".format(mean_squared_error(y_test, y_hat)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_test, y_hat)))
print("R2 entrenamiento: {}".format(r2_score(y_test, y_hat)))

#---------------------------------------------------------------------------

tun_svm_MSE= RandomizedSearchCV(m_svm,param_distributions=param_grid_svm,n_iter=3,scoring="neg_mean_squared_error")
tun_svm_MSE.fit(x_train,y_train)
y_pred = tun_svm_MSE.predict(x_train)


#Exactitud de modelo
print("Accuracy (Train): "+ str(tun_svm_MSE.score(x_train,y_train)*100) + "%")
#Metricas de desempeño de entrenamiento
print("MSE entrenamiento: {}".format(mean_squared_error(y_train, y_pred)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_train, y_pred)))
print("R2 entrenamiento: {}".format(r2_score(y_train, y_pred)))


#Predicciones sobre el conjunto de test
y_hat = tun_svm_MSE.predict(x_test)
#Exactitud de modelo
print("Accuracy (Test): "+ str(tun_svm_MSE.score(x_test,y_test)*100) + "%")
#Metricas de desempeño de validación
print("MSE entrenamiento: {}".format(mean_squared_error(y_test, y_hat)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_test, y_hat)))
print("R2 entrenamiento: {}".format(r2_score(y_test, y_hat)))

#---------------------------------------------------------------------------

tun_svm_MAE = RandomizedSearchCV(m_svm,param_distributions=param_grid_svm,n_iter=3,scoring="neg_mean_absolute_error")
tun_svm_MAE.fit(x_train,y_train)
y_pred = tun_svm_MAE.predict(x_train)


#Exactitud de modelo
print("Accuracy (Train): "+ str(tun_svm_MAE.score(x_train,y_train)*100) + "%")
#Metricas de desempeño de entrenamiento
print("MSE entrenamiento: {}".format(mean_squared_error(y_train, y_pred)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_train, y_pred)))
print("R2 entrenamiento: {}".format(r2_score(y_train, y_pred)))


#Predicciones sobre el conjunto de test
y_hat = tun_svm_MAE.predict(x_test)
#Exactitud de modelo
print("Accuracy (Test): "+ str(tun_svm_MAE.score(x_test,y_test)*100) + "%")
#Metricas de desempeño de validación
print("MSE entrenamiento: {}".format(mean_squared_error(y_test, y_hat)))
print("MAE entrenamiento: {}".format(mean_absolute_error(y_test, y_hat)))
print("R2 entrenamiento: {}".format(r2_score(y_test, y_hat)))

#--------------------------------------------------------------------------

###MODELO 3
#Definicion de Arquitectura

def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (ss_res / (ss_tot + K.epsilon()))

fa="tanh"
ann3=keras.models.Sequential([ 
    #keras.layers.Dense(128,activation=fa), #se quita una capa, se comprueba que esto afecta el accuracy del modelo
    keras.layers.Dense(64,activation=fa),
    keras.layers.Dense(32,activation=fa),
    keras.layers.Dense(1,activation="softmax") 
])

# Construir el modelo
ann3.build(input_shape=(None, x_train.shape[1]))

print("Parametros de Modelo 3: ",ann3.count_params()) 
lr=0.001  
optim=keras.optimizers.Adam(lr)
ann3.compile(optimizer=optim,loss="mean_squared_error" ,metrics=["mean_absolute_error"])
##Ajuste del modelo
ann3.fit(x_train,y_train,epochs=30,validation_data=(x_test,y_test),batch_size=20) 
# se disminye el batch_size, buscnado una mejora en los valores mas pequeños de las 
# metricas, por esto mismo, se aumentan los epochs, para dar mas espacio a buscar las 
# mejoras