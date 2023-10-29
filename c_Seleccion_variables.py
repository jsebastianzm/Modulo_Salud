import a_funciones as a
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor ##Ensamble con bagging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df_resultado = pd.read_csv('datos\datos_resultado.csv')


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


#mean_absolute_error, mean_squared_error, r2_score
rmse_mean=a.medir_modelos(modelos,"neg_root_mean_squared_error",X2,y,6)
rmse_absoluto=a.medir_modelos(modelos,'neg_median_absolute_error',X2,y,6)
rmse_r2=a.medir_modelos(modelos,"r2",X2,y,6)


rmse=pd.concat([rmse_mean, rmse_absoluto, rmse_r2],axis=1)
rmse.columns=['mean', 'absoluto', 'r2',
       'mean_Sel', 'absoluto_sel', 'r2_sel']

rmse_r2.plot(kind='box') #### gráfico para modelos todas las varibles
rmse_absoluto.plot(kind='box') ### gráfico para modelo variables absoluto
rmse_mean.plot(kind='box') ### mean
rmse.plot(kind='box') ### gráfico para modelos sel y todas las variables

rmse.mean() ### medias de mape