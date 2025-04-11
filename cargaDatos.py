import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# --- Configuración Inicial ---
FILE_PATH = 'Ab19selec.xlsx'
TARGET_VAR = 'CONSUMO ESPECIFICO CLO2'
SEED = 2022
TEST_SIZE = 0.20

# --- Función para limpieza de valores numéricos ---
def clean_numeric_value(value):
    try:
        if isinstance(value, str):
            # Reemplaza comas por puntos y elimina caracteres no numéricos
            value = value.replace(',', '.').strip()
            value = ''.join(c for c in value if c.isdigit() or c == '.')
            return float(value) if value else np.nan
        return float(value)
    except:
        return np.nan

# --- 1. Carga de Datos Robusta ---
print("\n=== CARGA DE DATOS ===")
try:
    # Carga inicial sin conversión forzada
    df = pd.read_excel(
        FILE_PATH,
        header=0,
        parse_dates=[0],
        usecols=lambda x: not x.startswith('Unnamed'),
        engine='openpyxl'
    )
    
    # Limpieza especial de la columna objetivo
    print(f"\nConvirtiendo columna objetivo '{TARGET_VAR}'...")
    df[TARGET_VAR] = df[TARGET_VAR].apply(clean_numeric_value)
    
    # Verificación
    print(f"\nMuestra de valores convertidos:")
    print(df[TARGET_VAR].head())
    print(f"\nValores nulos: {df[TARGET_VAR].isna().sum()}")
    
    # Establecer índice temporal si es datetime
    if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        df.set_index(df.columns[0], inplace=True)
        print("\nÍndice convertido a datetime")
    
    print(f"\n✅ Datos cargados correctamente. Dimensiones: {df.shape}")
    
except Exception as e:
    print(f"\n❌ Error crítico: {str(e)}")
    # Guardar diagnóstico
    if 'df' in locals():
        df.head(20).to_csv('error_diagnostico.csv')
        print("Se guardó muestra problemática en 'error_diagnostico.csv'")
    exit()

# --- 2. Generación de Metadatos ---
def generar_metadatos(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    constant_cols = numeric_cols[df[numeric_cols].nunique() == 1].tolist()
    
    metadata = {
        "missing_values": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "constant_columns": constant_cols,
        "numeric_columns": numeric_cols.tolist(),
        "target_stats": {
            "mean": df[TARGET_VAR].mean(),
            "std": df[TARGET_VAR].std(),
            "min": df[TARGET_VAR].min(),
            "max": df[TARGET_VAR].max()
        },
        "shape": list(df.shape),
        "columns": df.columns.tolist()
    }
    with open('metadatos.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False, default=str)
    return metadata

print("\n=== GENERANDO METADATOS ===")
metadata = generar_metadatos(df)
print("✅ Metadatos guardados en 'metadatos.json'")

# --- 3. Visualizaciones Integradas ---
print("\n=== GENERANDO VISUALIZACIONES ===")
plt.figure(figsize=(18, 6))

# Histograma + KDE
plt.subplot(1, 3, 1)
sns.histplot(df[TARGET_VAR].dropna(), kde=True, color='skyblue')
plt.title(f'Distribución de {TARGET_VAR}')
plt.xlabel('Consumo (kg/ADt)')
plt.grid(True, alpha=0.3)

# Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(y=df[TARGET_VAR], color='orange')
plt.title('Diagrama de Cajas')
plt.grid(True, alpha=0.3)

# Serie Temporal (si existe índice datetime)
if isinstance(df.index, pd.DatetimeIndex):
    plt.subplot(1, 3, 3)
    df[TARGET_VAR].plot(color='green')
    plt.title('Evolución Temporal')
    plt.xlabel('Fecha')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exploracion_inicial.png', dpi=300, bbox_inches='tight')
print("✅ Gráficos guardados en 'exploracion_inicial.png'")

# --- 4. Guardado de Datos Procesados ---
df.to_csv('datos_procesados.csv', index=True if isinstance(df.index, pd.DatetimeIndex) else False)
print("\n=== RESUMEN FINAL ===")
print(f"• Variables numéricas: {len(metadata['numeric_columns'])}")
print(f"• Columnas constantes: {metadata['constant_columns']}")
print(f"• Valores faltantes en target: {df[TARGET_VAR].isna().sum()}")
print(f"\n🎉 ¡Archivo cargaDatos.py ejecutado con éxito! 🎉")