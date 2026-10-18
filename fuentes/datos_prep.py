"""
datos_prep.py
-------------
Módulo de ingestión, auditoría, limpieza, estandarización y preprocesamiento
para el dataset de Telco Customer Churn.
Proyecto: MLOps & Experiment Tracking con MLflow
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga el dataset crudo desde un archivo CSV.
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"El archivo no existe en la ruta: {ruta_archivo}")
    df = pd.read_csv(ruta_archivo)
    return df


def auditar_datos(df: pd.DataFrame) -> dict:
    """
    Ejecuta una auditoría de calidad de datos identificando:
    - Dimensiones
    - Conteo de duplicados
    - Conteo de nulos y cadenas vacías por columna
    - Cardinalidad y tipos de datos
    """
    auditoria = {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "duplicados": int(df.duplicated().sum()),
        "nulos_por_columna": df.isnull().sum().to_dict(),
        "espacios_en_blanco": {col: int((df[col].astype(str).str.strip() == '').sum()) for col in df.columns},
        "tipos_datos": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    return auditoria


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las reglas de negocio y limpieza técnica:
    1. Remueve la columna de identificador 'customerID' (sin valor predictivo).
    2. Convierte 'TotalCharges' a numérico. Los 11 registros con espacios en blanco 
       corresponden a clientes con tenure == 0 (antigüedad cero meses), por lo que
       su valor acumulado se imputa como 0.0 según regla de negocio.
    3. Convierte 'SeniorCitizen' a tipo entero/categórico estándar.
    4. Estandariza la variable objetivo 'Churn' a binario (1: Churn/Abandono, 0: Retención).
    5. Normaliza cadenas de texto en columnas categóricas.
    """
    df_clean = df.copy()

    # 1. Eliminar identificador único
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])

    # 2. Manejo de TotalCharges y regla de negocio para tenure == 0
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'])
    
    # Imputación guiada por regla de negocio: si tenure == 0, TotalCharges = 0.0
    condicion_tenure_cero = (df_clean['tenure'] == 0) & (df_clean['TotalCharges'].isna())
    df_clean.loc[condicion_tenure_cero, 'TotalCharges'] = 0.0
    
    # Si quedaran otros nulos residuales, imputar con la mediana
    if df_clean['TotalCharges'].isna().sum() > 0:
        mediana_tc = df_clean['TotalCharges'].median()
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(mediana_tc)

    # 3. Codificación de variable objetivo 'Churn'
    if 'Churn' in df_clean.columns and df_clean['Churn'].dtype == 'object':
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # 4. Asegurar tipos numéricos consistentes
    df_clean['tenure'] = df_clean['tenure'].astype(int)
    df_clean['MonthlyCharges'] = df_clean['MonthlyCharges'].astype(float)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].astype(float)
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(int)

    return df_clean


def guardar_datos(df: pd.DataFrame, ruta_destino: str) -> None:
    """
    Guarda el dataframe limpio en formato CSV asegurando la existencia del directorio.
    """
    directorio = os.path.dirname(ruta_destino)
    if directorio and not os.path.exists(directorio):
        os.makedirs(directorio, exist_ok=True)
    df.to_csv(ruta_destino, index=False)
    print(f"Dataset guardado exitosamente en: {ruta_destino}")


def separar_x_y(df: pd.DataFrame, col_target: str = 'Churn'):
    """
    Separa variables independientes (X) y variable objetivo (y).
    """
    if col_target not in df.columns:
        raise ValueError(f"La columna objetivo '{col_target}' no se encuentra en el dataframe.")
    X = df.drop(columns=[col_target])
    y = df[col_target]
    return X, y


def obtener_definicion_columnas():
    """
    Retorna la lista de columnas numéricas y categóricas para el pipeline.
    """
    columnas_numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']
    columnas_categoricas = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    return columnas_numericas, columnas_categoricas


def crear_preprocesador(columnas_numericas, columnas_categoricas):
    """
    Construye un ColumnTransformer de scikit-learn con:
    - StandardScaler para columnas numéricas continuas.
    - OneHotEncoder para columnas categóricas nominales (drop='first' para evitar colinealidad).
    """
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), columnas_categoricas)
        ],
        remainder='passthrough'
    )
    return preprocesador


if __name__ == '__main__':
    # Script ejecutable para pruebas y preprocesamiento automático
    ruta_ini = os.path.join('datos', 'datos_ini', 'Telco-Customer-Churn.csv')
    ruta_limp = os.path.join('datos', 'datos_limp', 'Telco-Customer-Churn-Clean.csv')
    
    print("=== Módulo datos_prep: Iniciando proceso de limpieza ===")
    df_raw = cargar_datos(ruta_ini)
    auditoria = auditar_datos(df_raw)
    print(f"Auditoría inicial: {auditoria['filas']} filas, {auditoria['columnas']} columnas.")
    print(f"Espacios en blanco detectados en TotalCharges: {auditoria['espacios_en_blanco'].get('TotalCharges', 0)}")
    
    df_limpio = limpiar_datos(df_raw)
    guardar_datos(df_limpio, ruta_limp)
    print(f"Limpieza completada. Shape final: {df_limpio.shape}")
    print(f"Tasa de abandono (Churn=1): {df_limpio['Churn'].mean():.4f}")

