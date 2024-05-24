from itertools import cycle
import matplotlib.pyplot as plt

from sklearn.calibration import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_curve
import time

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
    23: "none"
}

def entrenar_modelo(X_train, y_train):
    # Create the KNN model
    modelo = KNeighborsClassifier(n_neighbors=7)
    # Train the model with the training data
    modelo.fit(X_train, y_train)
    return modelo

def guardar_modelo(modelo, nombre):
    # Guardamos el modelo entrenado para poder usarlo en otro lado después
    joblib.dump(modelo, nombre)

def probar_modelo(modelo, X_test, y_test, X_train, y_train):
    # # Con el dataset de prueba que viene como parámetro se hace una predicción
    # # usando el modelo entrenado previamente y obtener el accuracy
    y_pred = modelo.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # y_pred_train = modelo.predict(X_train)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # y_scores = modelo.predict_proba(X_test)
    
    # # CLASSIFICATION REPORT
    # classification_repor = classification_report(y_test, y_pred)
    # print("Classification Report:")
    # print(classification_repor)
    
    # # PRECISIÓN
    # print("Precisión:", accuracy)
    # print(f'Train score {accuracy_score(y_train, y_pred_train)}')
    # print(f'Test score {accuracy}')
    
    # # F1-SCORE
    # print("F1-Score:", f1)
    # print(f'Train score {f1_score(y_train, y_pred_train, average="weighted")}')
    # print(f'Test score {f1}')
    
    # # Este trozo de código está para ver qué técnicas detecta, creadno un dataframe de resultados 
    # # con 3 columnas: La técnica correcta, la técnica predicha y si la predicción es correcta 
    # # (si coinciden las otras dos columnas)
    
    # acertados = []
    # df_resultados = pd.DataFrame({'Real': y_test, 'Prediccion': y_pred})
    # df_resultados['Correcto'] = df_resultados['Real'] == df_resultados['Prediccion']
    # for index, row in df_resultados.iterrows():
    #     if row['Correcto'] and row['Prediccion'] not in acertados:
    #         acertados.append(row['Prediccion'])
    # # print(len(y_pred))
    # # print(len(y_test))
    # print(acertados)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Obtener las etiquetas de clase únicas
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))

    # Definir etiquetas de clase
    class_names = [f'{clases_dict[label]}' for label in unique_labels]

    # Configurar el tamaño de la figura en función del número de clases
    num_classes = len(unique_labels)
    plt.figure(figsize=(1.5*num_classes, 1*num_classes))

    # Crear la matriz de confusión
    sns.set_theme(font_scale=2.05) # Escala de fuente para mejorar la legibilidad
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=False, 
                xticklabels=class_names, yticklabels=class_names)

    # Añadir etiquetas y título
    plt.title("Matriz de Confusión")
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    
    # Mostrar la matriz de confusión
    # plt.show()
    # Guardar la matriz de confusión como un archivo .png
    plt.savefig('../Imagenes/Normal/KNN/matriz_confusion_KNN.png')

    
def trazar_ROC(model, X_test, y_test):
    clases = model.classes_
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
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
        mean_tpr += np.interp(all_fpr, false_positive_rate[i], true_positive_rate[i])
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
    plt.title('ROC de KNN Normal')
    plt.legend(loc="lower right", fontsize='small', ncol=2, title='Clases')

    # plt.show()
    plt.savefig('../Imagenes/Normal/KNN/ROC_KNN.png')

    


if __name__ == '__main__':
    ti = time.time()
    X_train = joblib.load('../Datos/Normal/X_train.pkl')
    X_test = joblib.load('../Datos/Normal/X_test.pkl')
    y_train = joblib.load('../Datos/Normal/y_train.pkl')
    y_test = joblib.load('../Datos/Normal/y_test.pkl')
    modelo = joblib.load('../Datos/Normal/modeloKNN.pkl')
    modelo = entrenar_modelo(X_train, y_train)
    guardar_modelo(modelo, '../Datos/Normal/modeloKNN.pkl')
    probar_modelo(modelo, X_test, y_test, X_train, y_train)
    trazar_ROC(modelo, X_test, y_test)
    tf = time.time()
    print("Tiempo de ejecución:", tf-ti)