# main.py
import os
from src.process_data import ejecutar_fase_2
from src.train_model import ejecutar_fase_3

# --- CONFIGURACIÓN CENTRALIZADA ---
CONFIG = {
    'fase_2': {
        'ruta_datos_raw': os.path.join('data', 'raw'),
        # Usaremos un nombre estándar para evitar confusiones
        'ruta_datos_procesados': os.path.join('data', 'processed', 'datos_limpios_para_modelo.csv'),
        
        # AJUSTA ESTAS COORDENADAS A TU CIUDAD (Ejemplo: CDMX)
        'geo_limites': {
            'lat_min': 19.0, 
            'lat_max': 19.9, 
            'lon_min': -99.4, 
            'lon_max': -98.8
        }
    },
    'fase_3': {
        # Debe coincidir con el archivo de salida de la Fase 2
        'ruta_datos_procesados': os.path.join('data', 'processed', 'datos_limpios_para_modelo.csv'),
        'ruta_modelo': os.path.join('models', 'sipid_model.joblib')
    }
}

if __name__ == "__main__":
    print("--- INICIANDO SISTEMA SIPID ---")
    
    # 1. Ejecutar Procesamiento (Limpieza + H3 + Guardar Texto)
    resultado = ejecutar_fase_2(CONFIG['fase_2'])
    
    if resultado == "exito":
        print("\n--- PASANDO A ENTRENAMIENTO ---")
        # 2. Ejecutar Entrenamiento (IA)
        ejecutar_fase_3(CONFIG['fase_3'])
    else:
        print("\n❌ La Fase 2 falló. No se puede continuar.")