import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from numpy import interp

clases_dict = {
    0: "Duplicate",
    1: "T1046",
    2: "T1059",
    3: "T1071",
    4: "T1112",
    5: "T1133",
    6: "T1136",
    7: "T1190",
    8: "T1203",
    9: "T1204",
    10: "T1210",
    11: "T1505",
    12: "T1546",
    13: "T1547",
    14: "T1548",
    15: "T1557",
    16: "T1566",
    17: "T1571",
    18: "T1587",
    19: "T1589",
    20: "T1590",
    21: "T1592",
    22: "T1595",
    23: "none",
    30: "Agrupadas"
}

def entrenar_modelo(X_train, y_train):
    # Creamos el modelo de Árbol de decisión de la librería
    modelo = DecisionTreeClassifier(ccp_alpha= 0.0333, criterion='entropy', max_depth=7, min_samples_leaf=1, min_samples_split=5)
    modelo.fit(X_train, y_train)
    return modelo


def guardar_modelo(modelo, nombre):
    # Guardamos el modelo entrenado para poder usarlo en otro lado después
    joblib.dump(modelo, nombre)


def probar_modelo(modelo, X_test, y_test, X_train, y_train):
    # Con el dataset de prueba que viene como parámetro se hace una predicción
    # usando el modelo entrenado previamente y obtener el accuracy
    clases = modelo.classes_
    y_pred = modelo.predict(X_test)
    y_pred_train = modelo.predict(X_train)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # CLASSIFICATION REPORT
    classification_repor = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(classification_repor)
    
    # PRECISIÓN
    print("Precisión:", accuracy)
    print(f'Train score {accuracy_score(y_train, y_pred_train)}')
    print(f'Test score {accuracy}')
    
    # F1-SCORE
    print("F1-Score:", f1)
    print(f'Train score {f1_score(y_train, y_pred_train, average="weighted")}')
    print(f'Test score {f1}')
    
    
    matriz_confusion_total = confusion_matrix(y_test, y_pred)

    # Obtener las etiquetas de clase únicas
    etiquetas = np.unique(np.concatenate((y_test, y_pred)))

    # Definir etiquetas de clase
    clases = [f'{clases_dict[etiqueta]}' for etiqueta in etiquetas]

    # Configurar el tamaño de la figura en función del número de clases
    num_clases = len(etiquetas)
    plt.figure(figsize=(1.5*num_clases, 1*num_clases))

    # Crear la matriz de confusión
    sns.set_theme(font_scale=1.9) # Escala de fuente para mejorar la legibilidad
    sns.heatmap(matriz_confusion_total, annot=True, fmt="d", cmap="Reds", cbar=False, 
                xticklabels=clases, yticklabels=clases)

    # Añadir etiquetas y título
    plt.title("Matriz de Confusión")
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")

    # Guardar la matriz de confusión como un archivo .png
    # plt.savefig('../Imagenes/Normal/DT/matriz_conf_DT.png')

    # Mostrar la matriz de confusión
    # plt.show()
    

def trazar_ROC(model, X_test, y_test):
    clases = model.classes_
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    print(y_test.values)
    y_ts = label_binarize(y_test.values, classes= clases)
    pred_prob = model.predict_proba(X_test)
    
    y_score = pred_prob
    print("y_ts shape:", y_ts.shape)
    print("y_score shape:", y_score.shape)
    print(model.classes_)
    # n_classes = y_ts.shape[1]
    n_classes = len(clases)

    # Definimos los diccionarios para guardar los valores de la true positive rate y false positive rate
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_area = dict()

    # Calculamos las ROC y AUC de cada clase
    for i in range(n_classes):
        false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_ts[:, i], y_score[:, i])
        roc_area[i] = auc(false_positive_rate[i], true_positive_rate[i])

    # Calculamos la micro-average ROC y AUC
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(y_ts.ravel(), y_score.ravel())
    roc_area["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    # Calculamos la macro-average ROC y AUC
    all_fpr = np.unique(np.concatenate([false_positive_rate[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, false_positive_rate[i], true_positive_rate[i])
    mean_tpr /= n_classes
    false_positive_rate["macro"] = all_fpr
    true_positive_rate["macro"] = mean_tpr
    roc_area["macro"] = auc(false_positive_rate["macro"], true_positive_rate["macro"])

    # Graficamos las ROC
    plt.figure(figsize=(10, 8))
    lw = 2

    # Colores para representarlas
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue'])

    for i, clase, color in zip(range(n_classes), clases, colors):
        plt.plot(false_positive_rate[i], true_positive_rate[i], color=color, linewidth=lw,
            label=f'Clase {clases_dict[clase]} (area = {roc_area[i]:0.2f})')

    # Agregamos la micro-average ROC
    plt.plot(false_positive_rate["micro"], true_positive_rate["micro"], color='deeppink', linestyle=':', linewidth=lw,
            label=f'Micro-average (area = {roc_area["micro"]:0.2f})')

    # Agregamos la macro-average ROC
    plt.plot(false_positive_rate["macro"], true_positive_rate["macro"], color='navy', linestyle=':', linewidth=lw,
            label=f'Macro-average (area = {roc_area["macro"]:0.2f})')

    # Parámetros de la gráfica
    plt.plot([0, 1], [0, 1], 'k--', linewidth=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC de DT Normal')
    plt.legend(loc="lower right", fontsize='small', ncol=2, title='Clases')

   
    plt.savefig('../Imagenes/Normal/DT/ROC_DT.png')

    plt.show()


def trazar_arbol(modelo, nombre):
    from sklearn.tree import plot_tree
    plt.figure(figsize=(15, 15))
    plot_tree(modelo, filled=True, max_depth=2, feature_names=X_train.columns, class_names=True)
    plt.savefig(nombre)
    plt.show()


if __name__ == '__main__':
    ti = time.time()
    X_train = joblib.load('../Datos/Normal/X_train.pkl')
    X_test = joblib.load('../Datos/Normal/X_test.pkl')
    y_train = joblib.load('../Datos/Normal/y_train.pkl')
    y_test = joblib.load('../Datos/Normal/y_test.pkl')
    modelo = entrenar_modelo(X_train, y_train)
    guardar_modelo(modelo, '../Datos/Normal/modeloDT.pkl')
    probar_modelo(modelo, X_test, y_test, X_train, y_train)
    trazar_ROC(modelo, X_test, y_test)
    trazar_arbol(modelo, '../Imagenes/Normal/DT/arbolDT_normal.png')
    tf = time.time()
    print("Tiempo de ejecución:", tf-ti)
    