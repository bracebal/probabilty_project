# modelaEvalua.py - Versión definitiva funcional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import joblib

# --- Configuración ---
SEED = 2022
sns.set_style('whitegrid')  # Estilo compatible

# --- Carga de datos ---
print("\n=== CARGA DE DATOS PREPARADOS ===")
try:
    # Cargar manteniendo DataFrames
    X_train = pd.read_csv('preparados_X_train.csv', index_col=0)
    X_test = pd.read_csv('preparados_X_test.csv', index_col=0)
    y_train = pd.read_csv('preparados_y_train.csv', index_col=0).squeeze()
    y_test = pd.read_csv('preparados_y_test.csv', index_col=0).squeeze()
    
    # Obtener nombres de características
    feature_names = X_train.columns.tolist()
    
    print(f"• X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"• X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"• Features: {len(feature_names)} (Primeras 5: {feature_names[:5]})")

except Exception as e:
    print(f"Error al cargar datos: {str(e)}")
    exit()

# --- Diccionario de resultados ---
results = {}

# --- 1. Regresión Lineal (MCO) ---
print("\n=== 1. REGRESIÓN LINEAL (MCO) ===")
try:
    X_train_const = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_const).fit()
    
    # Evaluación
    X_test_const = sm.add_constant(X_test)
    y_pred = ols_model.predict(X_test_const)
    
    results['OLS'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'Model': ols_model
    }
    print(ols_model.summary())
    
except Exception as e:
    print(f"Error en OLS: {str(e)}")
    results['OLS'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None}

# --- 2. Stepwise Selection ---
print("\n=== 2. STEPWISE SELECTION ===")
try:
    if len(feature_names) > 1:
        sfs = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select='auto',
            direction='forward',
            cv=5,
            n_jobs=-1
        ).fit(X_train, y_train)

        selected_features = X_train.columns[sfs.get_support()].tolist()
        print(f"Features seleccionadas ({len(selected_features)}): {selected_features}")
        
        # Modelo final
        lr_sw = LinearRegression().fit(X_train[selected_features], y_train)
        y_pred = lr_sw.predict(X_test[selected_features])
        
        results['Stepwise'] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'Features': selected_features,
            'Model': lr_sw
        }
    else:
        print("Solo hay 1 feature - Saltando Stepwise")
        results['Stepwise'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None}
        
except Exception as e:
    print(f"Error en Stepwise: {str(e)}")
    results['Stepwise'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None}

# --- 3. Ridge Regression ---
print("\n=== 3. RIDGE REGRESSION ===")
try:
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    ridge_cv.fit(X_train, y_train)

    print(f"Mejor alpha: {ridge_cv.alpha_:.4f}")

    y_pred = ridge_cv.predict(X_test)
    results['Ridge'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'Alpha': ridge_cv.alpha_,
        'Model': ridge_cv
    }
except Exception as e:
    print(f"Error en Ridge: {str(e)}")
    results['Ridge'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None}

# --- 4. Lasso Regression ---
print("\n=== 4. LASSO REGRESSION ===")
try:
    lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 50), cv=5, max_iter=10000)
    lasso_cv.fit(X_train, y_train)

    print(f"Mejor alpha: {lasso_cv.alpha_:.4f}")
    print(f"Features seleccionadas: {(lasso_cv.coef_ != 0).sum()}")

    y_pred = lasso_cv.predict(X_test)
    results['Lasso'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'Alpha': lasso_cv.alpha_,
        'Features': X_train.columns[lasso_cv.coef_ != 0].tolist(),
        'Model': lasso_cv
    }
except Exception as e:
    print(f"Error en Lasso: {str(e)}")
    results['Lasso'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None}

# --- 5. Elastic Net ---
print("\n=== 5. ELASTIC NET ===")
try:
    enet_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1], cv=5, max_iter=10000)
    enet_cv.fit(X_train, y_train)

    print(f"Mejor alpha: {enet_cv.alpha_:.4f}")
    print(f"Mejor l1_ratio: {enet_cv.l1_ratio_:.2f}")
    print(f"Features seleccionadas: {(enet_cv.coef_ != 0).sum()}")

    y_pred = enet_cv.predict(X_test)
    results['ElasticNet'] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'Alpha': enet_cv.alpha_,
        'L1_ratio': enet_cv.l1_ratio_,
        'Features': X_train.columns[enet_cv.coef_ != 0].tolist(),
        'Model': enet_cv
    }
except Exception as e:
    print(f"Error en ElasticNet: {str(e)}")
    results['ElasticNet'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None}

# --- Comparación y guardado ---
print("\n=== RESULTADOS FINALES ===")
comparison = pd.DataFrame.from_dict({
    model: {
        'MSE': metrics['MSE'],
        'R2': metrics['R2'],
        'MAE': metrics['MAE'],
        'N_Features': len(metrics.get('Features', feature_names))
    } 
    for model, metrics in results.items()
}, orient='index')

comparison['RMSE'] = np.sqrt(comparison['MSE'])
comparison = comparison.sort_values('R2', ascending=False)

print(comparison)

# Guardar resultados
joblib.dump(results, 'resultados_modelos.pkl')
comparison.to_csv('comparacion_modelos.csv')

# Gráfico comparativo
plt.figure(figsize=(10, 6))
comparison['R2'].plot(kind='bar', color='skyblue')
plt.title('Comparación de R2 entre Modelos')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('comparacion_r2.png')
plt.close()

print("\n✔ ¡Proceso completado con éxito!")
print(f"• Mejor modelo: {comparison.index[0]} (R2={comparison.iloc[0]['R2']:.3f})")
print(f"• Resultados guardados en:")
print("  - resultados_modelos.pkl")
print("  - comparacion_modelos.csv")
print("  - comparacion_r2.png")