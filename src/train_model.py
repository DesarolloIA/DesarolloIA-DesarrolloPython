# src/train_model.py
import pandas as pd
import joblib
import os
import h3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def ejecutar_fase_3(config):
    print("Iniciando Fase 3: Entrenamiento...")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv(config['ruta_datos_procesados'])
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        return

    # Cargar Encoder Grid
    try:
        le_grid = joblib.load('models/grid_encoder.joblib')
    except:
        print("No se encontró models/grid_encoder.joblib. Ejecuta Fase 2.")
        return

    # IMPORTANTE: Seleccionamos SOLO las columnas numéricas para X
    X = df[['hora_dia', 'dia_semana', 'mes', 'modalidad_num']]
    y = df['grid_num']
    coords_reales = df[['latitud', 'longitud']] 

    # 2. Split
    print("Dividiendo datos (Train/Test)...")
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X, y, coords_reales, test_size=0.2, random_state=42
    )

    # 3. Entrenamiento
    print("Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=50,      
        max_depth=15,         
        n_jobs=1,             
        min_samples_split=10, 
        random_state=42,
        verbose=1             
    )
    
    try:
        model.fit(X_train, y_train)
        print("¡Entrenamiento completado!")
    except MemoryError:
        print("❌ Error de Memoria RAM. Reduce n_estimators.")
        return

    # 4. Evaluación y Guardado
    print("Evaluando precisión espacial...")
    y_pred = model.predict(X_test)
    
    ids_h3 = le_grid.inverse_transform(y_pred)
    pred_coords = [h3.cell_to_latlng(x) for x in ids_h3] 
    pred_lat, pred_lon = zip(*pred_coords)
    
    kms = haversine_np(c_test['latitud'], c_test['longitud'], np.array(pred_lat), np.array(pred_lon))
    
    print(f"Resultados - Error Promedio: {np.mean(kms):.2f} km")
    
    os.makedirs(os.path.dirname(config['ruta_modelo']), exist_ok=True)
    joblib.dump(model, config['ruta_modelo'])
    print(f"✅ Modelo guardado en: {config['ruta_modelo']}")