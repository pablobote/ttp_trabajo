from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## En este caso se van a agrupar las clases 19 a 21 en una sola clase, y se van a eliminar las clases con menos de 4 instancias
columns_to_read = ['duration', 'conn_state', 'history', 'local_orig', 'local_resp', 'missed_bytes',
                   'orig_bytes', 'orig_ip_bytes', 'orig_pkts', 'proto', 'resp_bytes', 'resp_ip_bytes',
                   'resp_pkts','src_ip_zeek', 'src_port_zeek', 'dest_ip_zeek', 'service', 'label_tactic',
                   'label_technique']
columns_to_read_sin_tactica = ['community_id', 'conn_state', 'duration', 'history',
       'src_ip_zeek', 'src_port_zeek', 'dest_ip_zeek', 'dest_port_zeek',
       'local_orig', 'local_resp', 'missed_bytes', 'orig_bytes',
       'orig_ip_bytes', 'orig_pkts', 'proto', 'resp_bytes', 'resp_ip_bytes',
       'resp_pkts', 'service', 'ts', 'uid', 'datetime',
       'label_technique']




def preprocesar_datos():
    df = pd.read_csv('../Datos/uwf_total.csv', usecols= columns_to_read_sin_tactica)
    
    for atributo in df.columns:
        le = LabelEncoder()
        df[atributo]= le.fit_transform(df[atributo])
        
    le_tecnica = LabelEncoder()

    df['label_technique'] = le_tecnica.fit_transform(df['label_technique'])
    # Eliminamos las calses 0 y 14
    df = df[~df['label_technique'].isin([0, 14])]
    # Y agrupamos la 19, 20 y 21 en una sola clase
    df['label_technique'].replace({19: 30, 20: 30, 21: 30}, inplace=True)
    
    # Eliminamos ahora las clases con menos de 4 instancias
    df = df.groupby('label_technique').filter(lambda x: len(x) > 4)
    return df

def dividir_datos(df):
    
    # Sacamos el numero de muestras de 18 y 23
    num_muestras_18 = df[df['label_technique'] == 18].shape[0]
    num_muestras_23 = df[df['label_technique'] == 23].shape[0]

    # Calculamos el número de muestras a eliminar
    num_eliminar_18 = int(num_muestras_18 * 0.4)
    num_eliminar_23 = int(num_muestras_23 * 0.4)

    # Obtenemos los índices de las muestras a eliminar
    indices_eliminar_18 = np.random.choice(df[df['label_technique'] == 18].index, size=num_eliminar_18, replace=False)
    indices_eliminar_23 = np.random.choice(df[df['label_technique'] == 23].index, size=num_eliminar_23, replace=False)

    # Eliminar las muestras de las clases 18 y 23
    df = df.drop(indices_eliminar_18)
    df = df.drop(indices_eliminar_23)

    # Obtener los datos y etiquetas equilibrados
    X = df.drop(columns=['label_technique'])  # características
    y = df['label_technique']  # etiquetas

    # Dividir los datos equilibrados en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    c_matrix = X_train.corr('kendall')

    c_features = set()
    c_features = {c_matrix.columns[i] for i, col in enumerate(c_matrix.columns) for j in range(i) if abs(c_matrix.iloc[i, j]) > 0.8}
    
    print("Las características eliminadas son: ", c_features)
    print('')          
    X_test.drop(labels=c_features, axis=1, inplace=True)
    X_train.drop(labels=c_features, axis=1, inplace=True)
    plt.show()

    X_train = joblib.dump(X_train, '../Datos/Balanceado/X_train.pkl')
    y_train = joblib.dump(y_train, '../Datos/Balanceado/y_train.pkl')
    X_test = joblib.dump(X_test, '../Datos/Balanceado/X_test.pkl')
    y_test = joblib.dump(y_test, '../Datos/Balanceado/y_test.pkl')
    print("Samples 18: ",df['label_technique'].value_counts()[18])
    print("Samples 23: ", df['label_technique'].value_counts()[23])
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = preprocesar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(df)
    print(df['label_technique'].value_counts(normalize=True) * 100)
    