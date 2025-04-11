# preparaDatos.py - Versión optimizada

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# --- Configuración ---
INPUT_FILE = 'datos_procesados.csv'
OUTPUT_PREFIX = 'preparados'
SEED = 2022
TEST_SIZE = 0.20

# --- Carga de datos ---
print("\n=== PREPARACIÓN DE DATOS ===")
df = pd.read_csv(INPUT_FILE, parse_dates=True, index_col=0)

# Cargar metadatos
with open('metadatos.json') as f:
    metadata = json.load(f)

target = 'CONSUMO ESPECIFICO CLO2'

# --- 1. Limpieza ---
print("\n1. LIMPIEZA DE DATOS")

# 1.1 Eliminar filas con target faltante
initial_size = len(df)
df = df.dropna(subset=[target])
print(f"Eliminadas {initial_size - len(df)} filas con target faltante")

# 1.2 Eliminar columnas no numéricas (excepto target)
numeric_cols = df.select_dtypes(include=np.number).columns
non_numeric = [col for col in df.columns if col not in numeric_cols and col != target]

if non_numeric:
    print(f"Eliminando {len(non_numeric)} columnas no numéricas: {non_numeric}")
    df = df.drop(columns=non_numeric)

# --- 2. División Train-Test ---
print("\n2. DIVISIÓN DE DATOS")
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=SEED
)

# --- 3. Escalado Seguro ---
print("\n3. ESCALADO DE CARACTERÍSTICAS")

# Verificar que haya columnas numéricas para escalar
numeric_features = X_train.select_dtypes(include=np.number).columns

if len(numeric_features) > 0:
    print(f"Escalando {len(numeric_features)} características numéricas...")
    
    # Versión robusta del escalado
    scaler = StandardScaler()
    
    # Asegurarse de que no hay NaN antes de escalar
    assert X_train[numeric_features].isna().sum().sum() == 0, "Hay NaN en los datos de entrenamiento"
    
    X_train.loc[:, numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])
    
    # Guardar parámetros del escalador
    scaling_metadata = {
        'features_escaladas': numeric_features.tolist(),
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    }
else:
    print("Advertencia: No hay características numéricas para escalar")
    scaling_metadata = {}

# --- 4. Guardado de Resultados ---
print("\n4. GUARDADO DE DATOS")

# Guardar metadatos de preprocesamiento
with open('metadatos_preprocesamiento.json', 'w') as f:
    json.dump({
        'features_finales': X_train.columns.tolist(),
        'target_stats': {
            'train_mean': float(y_train.mean()),
            'test_mean': float(y_test.mean())
        },
        'scaling': scaling_metadata
    }, f, indent=4)

# Guardar datos
X_train.to_csv(f'{OUTPUT_PREFIX}_X_train.csv')
X_test.to_csv(f'{OUTPUT_PREFIX}_X_test.csv')
y_train.to_csv(f'{OUTPUT_PREFIX}_y_train.csv', header=True)
y_test.to_csv(f'{OUTPUT_PREFIX}_y_test.csv', header=True)

print("\n✔ ¡Preprocesamiento completado con éxito!")
print(f"• Train: {X_train.shape[0]} muestras, {X_train.shape[1]} features")
print(f"• Test: {X_test.shape[0]} muestras")