# src/process_data.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import glob
import h3       
import joblib   

def ejecutar_fase_2(config):
    print("Iniciando Fase 2: Validación, Preparación y Grid H3...")

    directorio_entrada = config['ruta_datos_raw']
    archivo_salida = config['ruta_datos_procesados']
    GEO = config['geo_limites']

    # 1. CARGA MASIVA
    try:
        patron = os.path.join(directorio_entrada, "*.xlsx")
        archivos = glob.glob(patron)
        if not archivos: 
            print("❌ No se encontraron archivos Excel en data/raw.")
            return "fallo"
            
        lista_dfs = []
        for f in archivos:
            try:
                # Intenta leer, asumiendo hoja 'RoboAuto' (ajusta si es necesario)
                df_temp = pd.read_excel(f, sheet_name='RoboAuto')
                
                # Normalizar columnas a mayúsculas y sin espacios
                df_temp.columns = [c.strip().upper() for c in df_temp.columns]
                
                # Verificar columnas críticas
                cols_necesarias = ['FECHA DE LOS HECHOS', 'HORA DE LOS HECHOS', 'COORD X', 'COORD Y', 'MODALIDAD - DELITO']
                if all(c in df_temp.columns for c in cols_necesarias):
                    lista_dfs.append(df_temp)
                else:
                    print(f"⚠️ El archivo {f} no tiene las columnas requeridas.")
            except Exception as e:
                print(f"Error leyendo {f}: {e}")

        if not lista_dfs: 
            return "fallo"
            
        df = pd.concat(lista_dfs, ignore_index=True)
        print(f"   -> Registros cargados: {len(df)}")

    except Exception as e:
        print(f"Error carga global: {e}")
        return "fallo"

    # 2. LIMPIEZA
    print(">> Procesando fechas y coordenadas...")
    
    # Hora a string seguro
    df['hora_str'] = df['HORA DE LOS HECHOS'].apply(lambda x: str(x) if pd.notnull(x) else "00:00:00")
    
    # Datetime (dayfirst=True es clave para formatos latinos)
    df['datetime_hechos'] = pd.to_datetime(
        df['FECHA DE LOS HECHOS'].astype(str).str.split().str[0] + ' ' + df['hora_str'], 
        dayfirst=True,   
        errors='coerce'
    )

    df['anio'] = df['datetime_hechos'].dt.year
    
    # Coordenadas numéricas
    df = df.rename(columns={'COORD Y': 'latitud', 'COORD X': 'longitud'})
    df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
    df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
    
    df = df.dropna(subset=['latitud', 'longitud', 'datetime_hechos'])

    # Filtro Geográfico
    df = df[
        (df['latitud'].between(GEO['lat_min'], GEO['lat_max'])) &
        (df['longitud'].between(GEO['lon_min'], GEO['lon_max']))
    ].copy()

    if df.empty:
        print("❌ Todos los datos quedaron fuera de los límites geográficos.")
        return "fallo"

    # 3. GENERAR GRID H3
    print(">> Creando Hexágonos H3...")
    df['grid_id'] = df.apply(lambda row: h3.latlng_to_cell(row['latitud'], row['longitud'], 8), axis=1)

    # Encoder Grid
    le_grid = LabelEncoder()
    df['grid_num'] = le_grid.fit_transform(df['grid_id'])
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_grid, 'models/grid_encoder.joblib')

    # 4. FEATURES Y TEXTO
    df['hora_dia'] = df['datetime_hechos'].dt.hour
    df['dia_semana'] = df['datetime_hechos'].dt.dayofweek
    df['mes'] = df['datetime_hechos'].dt.month
    
    # --- CORRECCIÓN CRÍTICA: Limpiar texto ---
    df['modalidad'] = df['MODALIDAD - DELITO'].astype(str).str.strip()
    
    # Encoder Delito
    le_mod = LabelEncoder()
    df['modalidad_num'] = le_mod.fit_transform(df['modalidad'])
    joblib.dump(le_mod, 'models/modalidad_encoder.joblib')

    # 5. GUARDAR
    # Guardamos 'modalidad' (texto) para el Dashboard y 'modalidad_num' para la IA
    cols = [
        'latitud', 'longitud',
        'grid_num',
        'hora_dia', 'dia_semana', 'mes', 'anio',
        'modalidad_num', 'modalidad'
    ]
    
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    df[cols].to_csv(archivo_salida, index=False)
    
    print(f"✅ ¡ÉXITO! Datos guardados en: {archivo_salida}")
    return "exito"