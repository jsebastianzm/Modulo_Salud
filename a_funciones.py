#Funciones
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate


def eliminar_caracteres_codificados(texto):
    # Define un diccionario de mapeo de caracteres a reemplazar
    reemplazos = {
        "á" : "a",
        "é" : 'e',
        "í": "i",
        "ó": "o",
        "ú": "u",
        "Á": "A",  # Letra mayúscula con acento
        "É": "E",
        "Í": "I",
        "Ó“": "O",
        "Ú": "U",
        # Agrega más caracteres y sus reemplazos si es necesario
    }

    # Itera sobre el diccionario de reemplazos y realiza la sustitución
    try:
        for codificado, acentuado in reemplazos.items():
            texto = texto.replace(codificado, acentuado)
        return texto
    except AttributeError:
        # Si no es una cadena, devuelve el valor original
        return texto

### Función para conocer información de cada una de las variables del dataframe
def info_columns(df):
    for i in df.columns:
        print(i, df[i].unique())
    print("Tamaño del DataFrame:",df.shape)

def sel_variables1(modelos, X, y, threshold):
    
    var_names_ac = np.array([])

    for modelo in modelos:
        # Crear una instancia del modelo
        modelo.fit(X, y)

        # Crear un objeto SelectFromModel y ajustarlo
        sel = SelectFromModel(modelo, prefit=True, threshold=threshold)
        sel.fit(X, y)

        # Obtener las características seleccionadas
        var_names = X.columns[sel.get_support()]

        # Agregar las características seleccionadas a var_names_ac
        var_names_ac = np.append(var_names_ac, var_names)

    # Eliminar duplicados y obtener las características únicas seleccionadas
    var_names_ac = np.unique(var_names_ac)

    return var_names_ac

def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["random_forest","SVM"]
    return metric_modelos
