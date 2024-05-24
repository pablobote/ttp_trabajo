from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from imblearn.over_sampling import SMOTE

## Código para preprocesar el dataset

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
    tecnicas = df['label_technique'].values
    for atributo in df.columns:
        le = LabelEncoder()
        df[atributo] = le.fit_transform(df[atributo])

    le_tecnica = LabelEncoder()
    tecnicas_cod = le_tecnica.fit_transform(df['label_technique'])
     
    df = df.groupby('label_technique').filter(lambda x: len(x) > 4)
    return df

    

def dividir_datos(df):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = df.drop(columns=['label_technique'])  # características
    y = df['label_technique']  # etiquetas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    joblib.dump(X_train, '../Datos/Normal/X_train.pkl')
    joblib.dump(X_test, '../Datos/Normal/X_test.pkl') 
    joblib.dump(y_train, '../Datos/Normal/y_train.pkl')
    joblib.dump(y_test, '../Datos/Normal/y_test.pkl')
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = preprocesar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(df)
    print('Datos preprocesados y divididos correctamente')