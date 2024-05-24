import time
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def modeloDT_normal(X_train, y_train):
    
    # Aquí se ponen los parámetros a probar en formato grid. 
    param_grid = {
    'criterion': [ 'entropy'],
    'max_depth': np.arange(7, 10),
    'min_samples_split': np.arange(0, 6),
    'min_samples_leaf': np.arange(1, 2),
    }
    
    # GridSearchCV coge esos parámetros de antes y prueba el modelo que le pasamos en estimator, con los diferentes parámetros hasta que encuentre el mejor en base a la métrica (accuracy)
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    # Imprimimos los mejores hiperparámetros para hacernos una idea e ir combinándolos con otros en la siguiente ejcución
    print("****** MODELO DT NORMAL ******")
    print(f"Mejores hiperparámetros encontrados:{grid_search.best_params_}")
    
def modeloDT_balanceado(X_train, y_train):

    # Aquí se ponen los parámetros a probar en formato grid. 
    param_grid = {
    'ccp_alpha' : np.arange(0.13, 0.4, 0.05),
    'criterion': [ 'entropy'],
    'max_depth': np.arange(7, 9),
    'min_samples_split': np.arange(3, 5),
    'min_samples_leaf': np.arange(1, 2)
    }
    
    # GridSearchCV coge esos parámetros de antes y prueba el modelo que le pasamos en estimator, con los diferentes parámetros hasta que encuentre el mejor en base a la métrica (accuracy)
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    # Imprimimos los mejores hiperparámetros para hacernos una idea e ir combinándolos con otros en la siguiente ejcución
    print("****** MODELO DT BALANCEADO ******")
    print(f"Mejores hiperparámetros encontrados:{grid_search.best_params_}")

    
def modeloRF_normal(X_train, y_train):
    
    # Define the parameter values that should be searched
    param_grid = {
        'n_estimators': [10, 20, 50],
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(7, 8),
        'min_samples_split': np.arange(4, 5),
        'min_samples_leaf': [1]
    }

    # Instantiate the grid
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')

    # Fit the grid with data
    grid.fit(X_train, y_train)

    # View the optimal parameters
    print("****** MODELO RF NORMAL ******")
    print("Best parameters: ", grid.best_params_)

def modeloRF_balanceado(X_train, y_train):
    
    # Define the parameter values that should be searched
    param_grid = {
        'n_estimators': [10, 20, 50],
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(7, 8),
        'min_samples_split': np.arange(4, 5),
        'min_samples_leaf': [1]
    }

    # Instantiate the grid
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')

    # Fit the grid with data
    grid.fit(X_train, y_train)

    # View the optimal parameters
    print("****** MODELO RF BALANCEADO ******")
    print("Best parameters: ", grid.best_params_)
    
    
def modeloKNN_normal(X_train, y_train):
    
    ti = time.time()
    # Define the parameter values that should be searched
    param_grid = {
    'n_neighbors': [3, 5, 7],             # Número de vecinos a considerar
    } 

    # Instantiate the grid
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')

    # Fit the grid with data
    grid.fit(X_train, y_train)

    # View the optimal parameters
    print("****** MODELO KNN NORMAL ******")
    print("Best parameters: ", grid.best_params_)
    tf = time.time()
    print("Tiempo de ejecución:", tf-ti)

def modeloKNN_balanceado(X_train, y_train):
    
    ti = time.time()
    # Define the parameter values that should be searched
    param_grid = {
    'n_neighbors': [3, 5, 7],             # Número de vecinos a considerar
    } 

    # Instantiate the grid
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')

    # Fit the grid with data
    grid.fit(X_train, y_train)

    # View the optimal parameters
    print("****** MODELO KNN BALANCEADO ******")
    print("Best parameters: ", grid.best_params_)
    tf = time.time()
    print("Tiempo de ejecución:", tf-ti)
    
if __name__ == '__main__':
    X_train_normal = joblib.load('../Datos/X_train.pkl')
    y_train_normal = joblib.load('../Datos/y_train.pkl')
    
    X_train_balanceado = joblib.load('../Datos/Balanceado/X_train.pkl')
    y_train_balanceado = joblib.load('../Datos/Balanceado/y_train.pkl')

    # modeloDT_normal(X_train_normal, y_train_normal)
    # modeloDT_balanceado(X_train_balanceado, y_train_balanceado)
    # modeloRF_normal(X_train_normal, y_train_normal)
    # modeloRF_balanceado(X_train_balanceado, y_train_balanceado)
    # modeloKNN_normal(X_train_normal, y_train_normal)
    modeloKNN_balanceado(X_train_balanceado, y_train_balanceado)
